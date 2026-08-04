"""Safe, pre-extraction ZIP validation."""
from __future__ import annotations

import stat
import zipfile
from pathlib import Path, PurePosixPath

from preprocess.batch.config import ArchiveConfig
from preprocess.batch.models import ArchiveInspection
from preprocess.batch.provenance import sha256_file


class ZipArchiveValidator:
    """Validate archive integrity, layout, size limits and video presence."""

    def __init__(self, config: ArchiveConfig | None = None) -> None:
        self.config = config or ArchiveConfig()

    def validate(self, archive_path: Path) -> ArchiveInspection:
        archive_path = archive_path.expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        archive_size = archive_path.stat().st_size
        if archive_size < self.config.minimum_archive_bytes:
            raise ValueError(f"Archive is smaller than configured minimum: {archive_path}")

        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                if self.config.max_members is not None and len(infos) > self.config.max_members:
                    raise ValueError(f"Archive has too many members: {len(infos)}")
                self._validate_members(infos)
                uncompressed_size = sum(info.file_size for info in infos if not info.is_dir())
                if (
                    self.config.max_uncompressed_bytes is not None
                    and uncompressed_size > self.config.max_uncompressed_bytes
                ):
                    raise ValueError(
                        f"Uncompressed archive size exceeds limit: {uncompressed_size} > "
                        f"{self.config.max_uncompressed_bytes}"
                    )
                for info in infos:
                    if info.is_dir():
                        continue
                    if (
                        self.config.max_member_uncompressed_bytes is not None
                        and info.file_size > self.config.max_member_uncompressed_bytes
                    ):
                        raise ValueError(
                            f"ZIP member exceeds size limit: {info.filename} "
                            f"{info.file_size} > {self.config.max_member_uncompressed_bytes}"
                        )
                    if self.config.max_compression_ratio is not None:
                        ratio = (
                            float("inf")
                            if info.file_size > 0 and info.compress_size == 0
                            else (info.file_size / max(info.compress_size, 1))
                        )
                        if ratio > self.config.max_compression_ratio:
                            raise ValueError(
                                f"ZIP member compression ratio exceeds limit: {info.filename}"
                            )
                corrupt_member = archive.testzip()
                if corrupt_member is not None:
                    raise ValueError(f"ZIP CRC validation failed for {corrupt_member}")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP archive: {archive_path}") from exc

        members = tuple(info.filename for info in infos)
        root_names = {self._root_name(name) for name in members if name.strip()}
        if root_names != {self.config.expected_root_name}:
            raise ValueError(
                f"Expected one archive root {self.config.expected_root_name!r}, got {sorted(root_names)}"
            )
        video_members = tuple(
            info.filename
            for info in infos
            if not info.is_dir() and Path(info.filename).suffix.lower() in self.config.video_extensions
        )
        if not video_members:
            raise ValueError(f"No supported video files found below {self.config.expected_root_name}/")
        return ArchiveInspection(
            archive_path=archive_path,
            archive_size_bytes=archive_size,
            root_name=self.config.expected_root_name,
            members=members,
            video_members=video_members,
            uncompressed_size_bytes=uncompressed_size,
            archive_sha256=sha256_file(archive_path),
        )

    def _validate_members(self, infos: list[zipfile.ZipInfo]) -> None:
        normalized_names: set[str] = set()
        casefolded_names: set[str] = set()
        for info in infos:
            normalized = info.filename.replace("\\", "/")
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe ZIP member path: {info.filename}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed in ZIP archives: {info.filename}")
            if normalized in normalized_names:
                raise ValueError(f"Duplicate ZIP member path: {info.filename}")
            folded = normalized.casefold()
            if folded in casefolded_names:
                raise ValueError(f"Case-colliding ZIP member path: {info.filename}")
            normalized_names.add(normalized)
            casefolded_names.add(folded)

    @staticmethod
    def _root_name(member_name: str) -> str:
        return PurePosixPath(member_name.replace("\\", "/")).parts[0]
