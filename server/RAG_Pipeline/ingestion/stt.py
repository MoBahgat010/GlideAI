import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union, List
from rev_ai import apiclient, JobStatus

logger = logging.getLogger("ingestion.stt")

MEDIA_EXTENSIONS = {
    ".3gp", ".aac", ".aiff", ".avi", ".flac", ".m4a", ".m4v",
    ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".ogg",
    ".opus", ".wav", ".webm", ".wma", ".wmv",
}


def is_media_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


class RevAITranscriber:
    """Transcribes media files with Rev AI and emits chunker-compatible structured JSON."""

    def __init__(
        self,
        access_token: str | None,
        poll_seconds: int = 10,
        max_segment_seconds: int = 60,
    ):
        self.access_token = access_token
        self.poll_seconds = poll_seconds
        self.max_segment_seconds = max_segment_seconds

    def load(self, file_paths: list[str] | str) -> list:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        return [SimpleNamespace(page_content=self.transcribe_to_json(path)) for path in file_paths]

    def transcribe_to_json(self, file_path: str | Path) -> str:
        if not self.access_token:
            raise EnvironmentError("Missing required Rev AI access token.")

        media_path = Path(file_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        client = apiclient.RevAiAPIClient(self.access_token)
        logger.info("Submitting %s to Rev AI for transcription...", media_path.name)
        job = client.submit_job_local_file(str(media_path))
        job_id = self._field(job, "id")
        if not job_id:
            raise RuntimeError(f"Rev AI did not return a job id for {media_path}")

        self._wait_for_job(client, job_id)
        transcript_json = client.get_transcript_json(job_id)
        if isinstance(transcript_json, str):
            transcript_json = json.loads(transcript_json)

        kids = self._transcript_to_elements(transcript_json)
        return json.dumps(
            {
                "file name": media_path.name,
                "source_type": "video_transcript",
                "transcription_provider": "rev_ai",
                "rev_ai_job_id": job_id,
                "kids": kids,
            }
        )

    def _normalize_status(self, raw_status) -> str:
        value = getattr(raw_status, "value", raw_status)
        return str(value).lower()

    def _wait_for_job(self, client, job_id: str) -> None:
        while True:
            details = client.get_job_details(job_id)
            status = self._normalize_status(self._field(details, "status", ""))
            logger.debug("Rev AI job %s status: %s", job_id, status)

            if status == JobStatus.TRANSCRIBED:
                return
            if status == JobStatus.FAILED:
                failure = self._field(details, "failure") or self._field(details, "failure_detail")
                raise RuntimeError(f"Rev AI transcription job {job_id} failed: {failure}")

            time.sleep(self.poll_seconds)

    @staticmethod
    def _field(obj, name: str, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _transcript_to_elements(self, transcript_json: dict) -> list[dict]:
        elements: list[dict] = []
        current_words: list[str] = []
        start_ts: float | None = None
        end_ts: float | None = None

        def flush() -> None:
            nonlocal current_words, start_ts, end_ts
            text = "".join(current_words).strip()
            if not text:
                current_words = []
                start_ts = None
                end_ts = None
                return

            elements.append(
                {
                    "type": "transcript",
                    "content": text,
                    "start_time": start_ts or 0.0,
                    "end_time": end_ts or start_ts or 0.0,
                    "page number": 1,
                }
            )
            current_words = []
            start_ts = None
            end_ts = None

        for monologue in transcript_json.get("monologues", []):
            for element in monologue.get("elements", []):
                value = element.get("value", "")
                if not value:
                    continue

                ts = element.get("ts")
                element_end = element.get("end_ts", ts)
                if ts is not None:
                    ts = float(ts)
                    element_end = float(element_end)
                    if start_ts is None:
                        start_ts = ts
                    elif ts - start_ts >= self.max_segment_seconds:
                        flush()
                        start_ts = ts
                    end_ts = element_end

                if element.get("type") == "punct":
                    current_words.append(value)
                else:
                    if current_words:
                        current_words.append(" ")
                    current_words.append(value)

        flush()
        return elements
