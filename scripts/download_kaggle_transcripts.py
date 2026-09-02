"""Download Kaggle notebook output → copy transcript .jsonl → AIC2026_sample/transcripts/.

Usage:
  export KAGGLE_USERNAME=<user> KAGGLE_KEY=<api-key>
  python scripts/download_kaggle_transcripts.py                          # notebook mặc định
  python scripts/download_kaggle_transcripts.py --notebook my-notebook   # slug khác
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = REPO_ROOT / "AIC2026_sample" / "transcripts"
TMP_DIR = REPO_ROOT / ".tmp_kaggle_output"


def ensure_kaggle_cli() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("kaggle CLI chưa có → pip install kaggle ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)


def ensure_credentials(username: str) -> None:
    if not Path.home().joinpath(".kaggle", "kaggle.json").exists():
        key = os.environ.get("KAGGLE_KEY")
        if not key:
            sys.exit("Thiếu KAGGLE_KEY — export KAGGLE_USERNAME + KAGGLE_KEY trước khi chạy")
        kd = Path.home() / ".kaggle"
        kd.mkdir(exist_ok=True)
        kj = kd / "kaggle.json"
        kj.write_text(json.dumps({"username": username, "key": key}))
        kj.chmod(0o600)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"),
                    help="Kaggle username (mặc định KAGGLE_USERNAME)")
    ap.add_argument("--notebook", default="kaggle-whisper-transcripts",
                    help="Notebook slug trên Kaggle")
    ap.add_argument("--dest", default=str(DEST_DIR), help="Thư mục chứa .jsonl (mặc định AIC2026_sample/transcripts)")
    args = ap.parse_args()

    if not args.user:
        sys.exit("Thiếu Kaggle username — dùng --user hoặc export KAGGLE_USERNAME")

    ensure_kaggle_cli()
    ensure_credentials(args.user)

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Tải output notebook {args.user}/{args.notebook} ...")
    subprocess.run(
        ["kaggle", "notebooks", "output", f"{args.user}/{args.notebook}", "-p", str(TMP_DIR)],
        check=True,
    )

    zips = list(TMP_DIR.glob("*.zip"))
    if not zips:
        sys.exit("Không thấy file output .zip — chắc chắn đã chạy 'Save & Run All' trên Kaggle?")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for zf in zips:
        with zipfile.ZipFile(zf) as z:
            for name in z.namelist():
                if name.endswith(".jsonl"):
                    target = dest / Path(name).name
                    if target.exists():
                        skipped += 1
                        continue
                    with z.open(name) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    copied += 1
                    print(f"  OK   {target.name}")

    print(f"\nDone. copied={copied} skipped(exists)={skipped} → {dest}")
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
