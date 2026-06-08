from pathlib import Path


def default_sample_root(repo_root: Path) -> Path:
    candidates = (
        repo_root / "AIC2026_sample",
        repo_root.parent / "AIC2026_sample",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def sample_subdir(sample_root: Path, name: str) -> Path:
    outer = sample_root / name
    nested = outer / name
    if nested.is_dir():
        return nested
    return outer
