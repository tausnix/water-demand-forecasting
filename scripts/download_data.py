"""Download and verify the Siemens/KIT hourly water-demand series.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waterdemand import config, data, features  # noqa: E402


def main() -> int:
    path = data.download(force="--force" in sys.argv)
    raw = data.load_raw(path)
    report = data.audit(raw, path)
    out = data.save_audit(report)

    model = features.build_features(raw)

    print(f"\n[data] audit written to {out}")
    print(f"  raw rows            : {report['rows']:,}")
    print(f"  missing             : {report['missing']}")
    print(f"  usable rows (warm-up): {len(model):,}  (expected {config.CANONICAL_ROWS - config.WARMUP_HOURS:,})")
    print(f"  time span           : {model[config.TIMESTAMP].min()} -> {model[config.TIMESTAMP].max()}")
    print(f"  feature count       : {len(config.FEATURES)}")

    expected = config.CANONICAL_ROWS - config.WARMUP_HOURS
    if len(model) != expected:
        print(f"  !! row count mismatch: {len(model)} != {expected}", file=sys.stderr)
        return 1
    print("  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
