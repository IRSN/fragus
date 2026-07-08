"""
prepare_eval_dataset_graph.py
==============================

GraphRAG equivalent of prepare_eval_dataset.py.
Runs GraphRAGPipeline on a question bank and produces the input file
expected by rag_evaluation.py — same output format, same Excel columns.

Usage:
  uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py questions.xlsx \\
      --experiment-id graph_db_rrf_top25 \\
      --mode flat_rrf

Supported modes:
  flat_rrf        ANN (BGE-M3) + BM25 Lucene → RRF → cross-encoder → top-25
  hybrid_cypher   ANN → BFS graph traversal (1-hop or 2-hop)
  text2cypher     LLM-generated Cypher query
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.rag.graph_rag_pipeline import (
    CANDIDATES,
    FULLTEXT_INDEX_NAME,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URL,
    NEO4J_USER,
    RRF_K,
    SCHEMA_TOP_K,
    TOP_K,
    TOP_N,
    VECTOR_INDEX_NAME,
    GraphRAGPipeline,
)
from scripts.rag_evaluation.prepare_eval_dataset import (
    COL_GROUP_ID,
    COL_INTENT,
    COL_IS_PARAPHRASE,
    COL_QID,
    COL_QUESTION,
    COL_REF_FILES,
    COL_REFERENCE,
    _PROXY_PREFIX,
    _build_paraphrase_llm,
    _fill_rag_result,
    _generate_paraphrases,
    _is_paraphrase,
    load_full_contexts,
    parse_reference_files,
    save_eval_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_rag(rag: GraphRAGPipeline, q_id: str, question: str) -> dict | None:
    try:
        return rag.ask(question)
    except Exception as exc:
        tqdm.write(f"  [ERROR] q_id={q_id}: {exc}")
        return None


def _make_row(q_id: str, question: str, is_para: bool, *,
              intent: str, group_id: str, reference: str, ref_files: list) -> dict:
    return {
        "q_id":               q_id,
        "test_intent":        intent,
        "paraphrase_group_id": group_id,
        "is_paraphrase":      is_para,
        "user_input":         question,
        "response":           None,
        "reference":          reference,
        "retrieved_contexts": None,
        "reference_files":    json.dumps(ref_files, ensure_ascii=False),
        "cited_files":        None,
        "retrieved_files":    None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run GraphRAGPipeline on a question bank and produce rag_evaluation.py input."
    )
    ap.add_argument("input", help="Input .xlsx with the question bank")
    ap.add_argument("--experiment-id", dest="experiment_id", default=None)
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--paraphrase-variants", type=int, default=3, dest="paraphrase_variants")

    # Neo4j connection
    ap.add_argument("--neo4j-url",      default=NEO4J_URL)
    ap.add_argument("--neo4j-user",     default=NEO4J_USER)
    ap.add_argument("--neo4j-password", default=NEO4J_PASSWORD)
    ap.add_argument("--database",       default=NEO4J_DATABASE)

    # Pipeline mode
    ap.add_argument("--mode", default="flat_rrf",
                    choices=["flat_rrf", "hybrid_cypher", "text2cypher"])
    ap.add_argument("--no-prune-schema", action="store_true", dest="no_prune_schema",
                    help="Disable schema pruning entirely (passes full schema to the LLM — unsafe on large graphs).")
    ap.add_argument("--schema-top-k", type=int, default=SCHEMA_TOP_K, dest="schema_top_k",
                    help=f"Schema lines kept per query when embedding pruning is active (default: {SCHEMA_TOP_K}).")

    # flat_rrf params
    ap.add_argument("--vector-index",   default=VECTOR_INDEX_NAME, dest="vector_index")
    ap.add_argument("--fulltext-index", default=FULLTEXT_INDEX_NAME, dest="fulltext_index")
    ap.add_argument("--candidates",     type=int, default=CANDIDATES)
    ap.add_argument("--top-n",          type=int, default=TOP_N, dest="top_n")
    ap.add_argument("--rrf-k",          type=int, default=RRF_K, dest="rrf_k")
    ap.add_argument("--reranker-url",   default=None, dest="reranker_url")

    # hybrid_cypher params
    ap.add_argument("--top-k", type=int, default=TOP_K, dest="top_k")

    args = ap.parse_args()

    _results_dir = Path(__file__).parent / "data" / "results"
    _results_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = args.experiment_id or Path(args.input).stem
    output_path = _results_dir / f"eval_test_{experiment_id}.xlsx"
    n_variants = max(0, args.paraphrase_variants)

    # Load question bank
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df_in = pd.read_excel(args.input, sheet_name=sheet, nrows=args.nrows)
    print(f"Loaded {len(df_in)} rows from '{args.input}'")

    for col in (COL_QID, COL_QUESTION, COL_REFERENCE, COL_REF_FILES):
        if col not in df_in.columns:
            raise KeyError(f"Required column '{col}' not found. Available: {list(df_in.columns)}")

    in_para = (
        df_in[COL_IS_PARAPHRASE].apply(_is_paraphrase)
        if COL_IS_PARAPHRASE in df_in.columns
        else pd.Series(False, index=df_in.index)
    )
    originals_df = df_in[~in_para].copy()
    input_variants_by_group: dict[str, list[str]] = {}
    if COL_IS_PARAPHRASE in df_in.columns and COL_GROUP_ID in df_in.columns:
        for _, vrow in df_in[in_para].iterrows():
            gid = str(vrow.get(COL_GROUP_ID, "")).strip()
            if gid:
                input_variants_by_group.setdefault(gid, []).append(
                    str(vrow.get(COL_QUESTION, "")).strip()
                )

    # Resume: load previous output
    already_done: set[str] = set()
    original_done: set[str] = set()
    rows: dict[str, dict] = {}
    by_qid: dict[str, dict] = {}
    if output_path.exists():
        df_prev = pd.read_excel(output_path)
        for _, r in df_prev.iterrows():
            qid = str(r.get("q_id", "")).strip()
            by_qid[qid] = r.to_dict()
        rows = dict(by_qid)
        load_full_contexts(rows, output_path)  # restore full contexts from parquet sidecar
        existing_variant_count: dict[str, int] = {}
        for r in by_qid.values():
            if _is_paraphrase(r.get("is_paraphrase")):
                gid = str(r.get("paraphrase_group_id", "")).strip()
                if gid:
                    existing_variant_count[gid] = existing_variant_count.get(gid, 0) + 1
        for qid, r in by_qid.items():
            if not _is_paraphrase(r.get("is_paraphrase")) and pd.notna(r.get("response")):
                original_done.add(qid)
                if existing_variant_count.get(qid, 0) >= n_variants:
                    already_done.add(qid)
        if already_done:
            print(f"Resuming: {len(already_done)} groups fully done (skipped).")

    remaining = originals_df[
        ~originals_df[COL_QID].astype(str).str.strip().isin(already_done)
    ]
    if remaining.empty:
        print("All questions already answered. Nothing to do.")
        return

    # Init pipeline
    print(f"Initialising GraphRAGPipeline (mode={args.mode})...")
    pipeline_kwargs: dict = dict(
        neo4j_url=args.neo4j_url,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        database=args.database,
        mode=args.mode,
        vector_index_name=args.vector_index,
        prune_schema=not args.no_prune_schema,
        schema_top_k=args.schema_top_k,
    )
    if args.mode == "flat_rrf":
        pipeline_kwargs.update(
            fulltext_index_name=args.fulltext_index,
            candidates=args.candidates,
            top_n=args.top_n,
            rrf_k=args.rrf_k,
            reranker_url=args.reranker_url,
        )
    elif args.mode == "hybrid_cypher":
        pipeline_kwargs.update(top_k=args.top_k)

    rag = GraphRAGPipeline(**pipeline_kwargs)
    print("  Pipeline ready.")

    # Init paraphrase LLM if needed
    paraphrase_llm = None
    if n_variants > 0:
        needs_llm_variants = any(
            len(input_variants_by_group.get(str(r[COL_QID]).strip(), [])) < n_variants
            for _, r in remaining.iterrows()
        )
        if needs_llm_variants:
            print(f"Initialising paraphrase LLM...")
            paraphrase_llm = _build_paraphrase_llm(_PROXY_PREFIX)

    # Process questions
    errors = 0
    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Questions"):
        q_id      = str(row.get(COL_QID, "")).strip()
        question  = str(row.get(COL_QUESTION, "")).strip()
        reference = str(row.get(COL_REFERENCE, "")).strip()
        ref_files = parse_reference_files(row.get(COL_REF_FILES))
        intent    = str(row.get(COL_INTENT, "")).strip() if COL_INTENT in df_in.columns else ""
        group_id  = str(row.get(COL_GROUP_ID, q_id)).strip() or q_id

        row_kwargs = dict(intent=intent, group_id=group_id,
                          reference=reference, ref_files=ref_files)

        # Original question
        if q_id in original_done and q_id in by_qid:
            orig_row = dict(by_qid[q_id])
            orig_row["paraphrase_group_id"] = group_id
            orig_row["is_paraphrase"] = False
        else:
            orig_row = _make_row(q_id, question, False, **row_kwargs)
            if question:
                result = _run_rag(rag, q_id, question)
                if result:
                    _fill_rag_result(orig_row, result)
                else:
                    errors += 1
        rows[q_id] = orig_row

        # Variants
        existing_variant_qs = input_variants_by_group.get(group_id, [])
        missing = n_variants - len(existing_variant_qs)
        generated_qs: list[str] = []
        if missing > 0 and paraphrase_llm and question:
            generated_qs = _generate_paraphrases(paraphrase_llm, question, missing)
        all_variant_qs = existing_variant_qs + generated_qs

        for i, variant_q in enumerate(all_variant_qs[:n_variants], 1):
            var_id = f"{q_id}_v{i}"
            if var_id in by_qid and pd.notna(by_qid[var_id].get("response")):
                rows[var_id] = dict(by_qid[var_id])
            else:
                var_row = _make_row(var_id, variant_q, True, **row_kwargs)
                if variant_q:
                    result = _run_rag(rag, var_id, variant_q)
                    if result:
                        _fill_rag_result(var_row, result)
                rows[var_id] = var_row

        # Save after each question (resume-safe)
        save_eval_outputs(rows, output_path)

    rag.close()

    print(f"\nDone. {len(rows)} rows → {output_path}")
    if errors:
        print(f"  {errors} RAG errors (check logs).")


if __name__ == "__main__":
    main()
