"""Parse direct archive URLs from a line-oriented input file."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from preprocess.batch.models import ArchiveInput


class LinkListParser:
    """Turn one URL per line into deterministic archive identities."""

    def __init__(self, *, archive_prefix: str = "Videos_", archive_extension: str = ".zip") -> None:
        self.archive_prefix = archive_prefix
        self.archive_extension = archive_extension.lower()

    def parse(self, path: Path) -> list[ArchiveInput]:
        if not path.is_file():
            raise FileNotFoundError(f"Links file not found: {path}")

        requests: list[ArchiveInput] = []
        seen_urls: set[str] = set()
        seen_lots: set[str] = set()
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            request = self._parse_line(line, line_number)
            if request.url in seen_urls:
                raise ValueError(f"Duplicate URL at line {line_number}: {request.url}")
            if request.lot_id in seen_lots:
                raise ValueError(f"Duplicate lot_id at line {line_number}: {request.lot_id}")
            seen_urls.add(request.url)
            seen_lots.add(request.lot_id)
            requests.append(request)
        if not requests:
            raise ValueError(f"No archive URLs found in {path}")
        return requests

    def _parse_line(self, url: str, line_number: int) -> ArchiveInput:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Line {line_number}: expected an HTTP(S) URL")
        filename = unquote(PurePosixPath(parsed.path).name)
        if not filename:
            raise ValueError(f"Line {line_number}: URL has no archive filename")
        if not filename.lower().endswith(self.archive_extension):
            raise ValueError(
                f"Line {line_number}: expected {self.archive_extension} archive, got {filename!r}"
            )
        stem = filename[: -len(self.archive_extension)]
        lot_id = stem[len(self.archive_prefix) :] if stem.startswith(self.archive_prefix) else stem
        if not lot_id or lot_id in {".", ".."} or Path(lot_id).name != lot_id:
            raise ValueError(f"Line {line_number}: unsafe/empty lot id derived from {filename!r}")
        return ArchiveInput(url=url, archive_name=filename, lot_id=lot_id, line_number=line_number)
