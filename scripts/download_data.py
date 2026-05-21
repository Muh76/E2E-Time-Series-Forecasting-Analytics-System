"""
Download the Rossmann Store Sales dataset from Kaggle into data/raw/.

Automated path (recommended):
    pip install kaggle
    # Supply credentials via ~/.kaggle/kaggle.json OR KAGGLE_USERNAME + KAGGLE_KEY in .env
    python scripts/download_data.py

Manual fallback (no Kaggle CLI):
    python scripts/download_data.py --manual

The script validates that all required CSV files are present after download and
exits with a non-zero code if anything is missing, making it safe to use in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
COMPETITION = "rossmann-store-sales"
REQUIRED_FILES = ["train.csv", "store.csv", "test.csv", "sample_submission.csv"]
KAGGLE_URL = "https://www.kaggle.com/competitions/rossmann-store-sales/data"


def _load_dotenv() -> None:
    """Load .env from the project root so KAGGLE_* vars are visible."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        # Manually parse the file if python-dotenv isn't installed yet.
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _missing_files() -> list[str]:
    return [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]


def _kaggle_credentials_available() -> bool:
    """Return True when Kaggle credentials can be found."""
    # Env-var path (also set via .env)
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    # File path (~/.kaggle/kaggle.json)
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text())
            return bool(data.get("username") and data.get("key"))
        except Exception:
            pass
    return False


def print_manual_instructions() -> None:
    print(
        "\n"
        "Manual download instructions\n"
        "============================\n"
        "\n"
        "1. Sign in to Kaggle and accept the competition rules:\n"
        f"   {KAGGLE_URL}\n"
        "\n"
        "2. Download all competition files (train.csv, store.csv, test.csv,\n"
        "   sample_submission.csv) from the Data tab.\n"
        "\n"
        "3. Place the unzipped CSV files in:\n"
        f"   {RAW_DIR}\n"
        "\n"
        "   Expected layout:\n"
        "   data/raw/\n"
        "   ├── train.csv\n"
        "   ├── store.csv\n"
        "   ├── test.csv\n"
        "   └── sample_submission.csv\n"
        "\n"
        "Alternatively, install the Kaggle CLI and run:\n"
        "   pip install kaggle\n"
        "   # Set KAGGLE_USERNAME and KAGGLE_KEY in .env (see .env.example)\n"
        "   python scripts/download_data.py\n"
    )


def download_via_kaggle() -> bool:
    """
    Download and extract the competition dataset using the Kaggle Python API.
    Returns True on success, False on failure.
    """
    try:
        import kaggle  # noqa: F401 — triggers credential validation on import
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print(
            "ERROR: kaggle package not installed.\n"
            "       Run: pip install kaggle\n"
            "       Then re-run this script, or use --manual for instructions."
        )
        return False
    except Exception as exc:
        print(f"ERROR: Kaggle API initialisation failed: {exc}")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Rossmann Store Sales dataset to {RAW_DIR} …")
    try:
        api = KaggleApiExtended()
        api.authenticate()
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}")
        return False

    # Extract any ZIP archives dropped by the Kaggle CLI.
    for zip_path in RAW_DIR.glob("*.zip"):
        print(f"Extracting {zip_path.name} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    return True


def validate_files() -> bool:
    missing = _missing_files()
    if missing:
        print(f"ERROR: Missing files in {RAW_DIR}: {', '.join(missing)}")
        return False
    print(f"All required files present in {RAW_DIR}:")
    for f in REQUIRED_FILES:
        size_mb = (RAW_DIR / f).stat().st_size / 1_048_576
        print(f"  {f:<30} {size_mb:6.1f} MB")
    return True


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual download instructions and exit.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only check whether required files are present; do not download.",
    )
    args = parser.parse_args()

    if args.manual:
        print_manual_instructions()
        return 0

    if args.validate_only:
        return 0 if validate_files() else 1

    # Skip download if all files are already present.
    if not _missing_files():
        print("All required files already present. Nothing to download.")
        return 0 if validate_files() else 1

    if not _kaggle_credentials_available():
        print(
            "ERROR: Kaggle credentials not found.\n"
            "\n"
            "Option A — credential file:\n"
            "  Create ~/.kaggle/kaggle.json:\n"
            '  {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}\n'
            "\n"
            "Option B — environment variables:\n"
            "  Add to .env (see .env.example):\n"
            "  KAGGLE_USERNAME=your_username\n"
            "  KAGGLE_KEY=your_api_key\n"
            "\n"
            "Get your API key from https://www.kaggle.com/settings → API.\n"
            "Run with --manual for manual download instructions."
        )
        return 1

    if not download_via_kaggle():
        print("\nDownload failed. Run with --manual for manual download instructions.")
        return 1

    if not validate_files():
        print("\nDownload completed but some expected files are missing.")
        return 1

    print("\nData download complete. You can now run: python scripts/run_etl.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
