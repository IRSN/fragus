"""Local patch of DoclingServiceClient to pass through full HybridChunkerOptions.

The public client (docling 2.90.0) always creates HybridChunkerOptions() with the
default values in _submit_chunk_task, ignoring tokenizer/max_tokens/merge_peers.
This subclass overrides only that private method.
"""

from __future__ import annotations

from typing import Any

from docling.datamodel.service.chunking import (
    HierarchicalChunkerOptions,
    HybridChunkerOptions,
)
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.requests import HttpSourceRequest
from docling.datamodel.service.responses import TaskStatusResponse
from docling.datamodel.service.targets import InBodyTarget
from docling.service_client import DoclingServiceClient, ChunkerKind

SourceType = Any  # same as docling.service_client.client.SourceType


class PatchedDoclingServiceClient(DoclingServiceClient):
    """DoclingServiceClient that accepts HybridChunkerOptions/HierarchicalChunkerOptions."""

    def chunk(
        self,
        source: SourceType,
        chunker: ChunkerKind | HybridChunkerOptions | HierarchicalChunkerOptions,
        options: ConvertDocumentsRequestOptions | None = None,
    ):
        job = self.submit_chunk(source=source, chunker=chunker, options=options)
        return job.result(timeout=self._job_timeout)

    def submit_chunk(
        self,
        source: SourceType,
        chunker: ChunkerKind | HybridChunkerOptions | HierarchicalChunkerOptions,
        options: ConvertDocumentsRequestOptions | None = None,
    ):
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        initial_status = self._submit_chunk_task(
            source=source,
            chunker=chunker,
            options=resolved.options,
        )
        from docling.service_client.job import ConversionJob, _JobHandlers
        from docling.datamodel.service.responses import ChunkDocumentResponse
        from datetime import datetime, timezone

        handlers = _JobHandlers[ChunkDocumentResponse](
            poll=self._poll_task_status,
            watch=self._watch_task_updates,
            wait=self._wait_for_terminal_status,
            fetch_result=lambda task_id, last_status: self._fetch_chunk_result(
                task_id=task_id,
                last_status=last_status,
            ),
        )
        return ConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    def _submit_chunk_task(
        self,
        source: SourceType,
        chunker: ChunkerKind | HybridChunkerOptions | HierarchicalChunkerOptions,
        options: ConvertDocumentsRequestOptions,
    ) -> TaskStatusResponse:
        chunking_options: HybridChunkerOptions | HierarchicalChunkerOptions
        if isinstance(chunker, (HybridChunkerOptions, HierarchicalChunkerOptions)):
            chunking_options = chunker
        elif chunker == ChunkerKind.HYBRID:
            chunking_options = HybridChunkerOptions()
        else:
            chunking_options = HierarchicalChunkerOptions()
        chunker_kind_value = chunking_options.chunker.value

        if isinstance(source, str):
            self._validate_http_source(source)
            payload = {
                "convert_options": options.model_dump(mode="json", exclude_none=True),
                "chunking_options": chunking_options.model_dump(mode="json", exclude_none=True),
                "sources": [
                    HttpSourceRequest(url=source, headers={}).model_dump(mode="json")
                ],
                "include_converted_doc": False,
                "target": InBodyTarget().model_dump(mode="json"),
                "callbacks": [],
            }
            response = self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker_kind_value}/source/async",
                json=payload,
            )
        else:
            files = self._source_to_upload_files(source)
            data: dict[str, Any] = {
                f"convert_{key}": value
                for key, value in options.model_dump(mode="json", exclude_none=True).items()
            }
            chunk_payload = chunking_options.model_dump(mode="json", exclude_none=True)
            chunk_payload.pop("chunker", None)
            data.update({f"chunking_{key}": value for key, value in chunk_payload.items()})
            data["include_converted_doc"] = False
            data["target_type"] = "inbody"
            response = self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker_kind_value}/file/async",
                data=data,
                files=files,
            )

        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Chunk task submission failed.")
        return TaskStatusResponse.model_validate_json(response.text)
