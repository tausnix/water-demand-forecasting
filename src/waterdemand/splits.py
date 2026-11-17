"""Chronological splitting.

No shuffling, ever. The test block is materialised once and must not be read
until ``reports/selection.json`` exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from . import config


@dataclass(frozen=True)
class SplitInfo:
    rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    n = len(frame)
    i1 = int(n * config.TRAIN_FRAC)
    i2 = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))

    train = frame.iloc[:i1].copy()
    validation = frame.iloc[i1:i2].copy()
    test = frame.iloc[i2:].copy()

    info = SplitInfo(
        rows=n,
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_end=train[config.TIMESTAMP].iloc[-1].isoformat(),
        validation_start=validation[config.TIMESTAMP].iloc[0].isoformat(),
        validation_end=validation[config.TIMESTAMP].iloc[-1].isoformat(),
        test_start=test[config.TIMESTAMP].iloc[0].isoformat(),
        test_end=test[config.TIMESTAMP].iloc[-1].isoformat(),
    )

    assert len(train) + len(validation) + len(test) == n
    assert train[config.TIMESTAMP].max() < validation[config.TIMESTAMP].min()
    assert validation[config.TIMESTAMP].max() < test[config.TIMESTAMP].min()

    return train, validation, test, info


def as_dict(info: SplitInfo) -> dict:
    return asdict(info)
