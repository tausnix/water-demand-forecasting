"""Dataset acquisition, integrity checking and the raw-data audit.

The project ships no synthetic fallback. If the CSV is missing, it is fetched
from Zenodo; if it is present but does not match the pinned hash, we refuse to
continue rather than training on an unknown series.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from . import config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> Path:
    """Fetch the canonical series from Zenodo and verify its checksum."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    dest = config.CANONICAL_CSV

    if dest.exists() and not force:
        actual = sha256(dest)
        if actual == config.CANONICAL_SHA256:
            print(f"[data] cached, sha256 OK: {dest}")
            return dest
        print(f"[data] checksum mismatch on cached file, re-downloading ({actual[:12]}...)")

    if not config.ZENODO_URL.startswith("https://"):
        raise ValueError("refusing non-HTTPS download")
    print(f"[data] downloading {config.ZENODO_URL}")
    with urllib.request.urlopen(config.ZENODO_URL, timeout=120) as response:
        payload = response.read()
    dest.write_bytes(payload)

    actual = sha256(dest)
    if actual != config.CANONICAL_SHA256:
        raise RuntimeError(
            "dataset checksum mismatch\n"
            f"  expected {config.CANONICAL_SHA256}\n"
            f"  actual   {actual}"
        )
    print(f"[data] downloaded and verified ({len(payload):,} bytes)")
    return dest


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the canonical CSV exactly as published (Date, Time, Consumption, ...)."""
    path = path or config.CANONICAL_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.py first")
    return pd.read_csv(path)


def audit(frame: pd.DataFrame, source: Path | None = None) -> dict:
    """Record shape, missingness and provenance. Written to reports/data_audit.json."""
    report = {
        "source": "Hourly Water Demand of a Mixed District Recorded by Supplier",
        "zenodo_record": config.ZENODO_RECORD,
        "doi": config.ZENODO_DOI,
        "synthetic": False,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing": {k: int(v) for k, v in frame.isna().sum().items()},
        "date_min": str(frame["Date"].iloc[0]),
        "date_max": str(frame["Date"].iloc[-1]),
        "consumption_describe": {
            k: float(v)
            for k, v in frame["Consumption"].describe().items()
        },
    }
    if source is not None:
        report["sha256"] = sha256(source)
        report["bytes"] = source.stat().st_size
    return report


def save_audit(report: dict) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "data_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out
