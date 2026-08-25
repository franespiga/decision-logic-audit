"""
Access to the frozen market data shared by every policy in the comparison.

One HDF5 file supplies both families: ``technical_indicators`` (seven indicators
for six equity indices) and ``instruments`` (Open/High/Low/Close/Volume for the
six indices and for the cash instrument). The merged frame is built exactly as
the training environments built it -- indicators joined to prices on the trading
calendar -- so that a policy's observation columns can be selected by name
without re-deriving anything.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class MarketData:
    path: str
    prices: pd.DataFrame
    indicators: pd.DataFrame
    merged: pd.DataFrame

    @property
    def indicator_columns(self) -> List[Tuple[str, str]]:
        return [tuple(c) for c in self.indicators.columns]

    @property
    def price_columns(self) -> List[Tuple[str, str]]:
        return [tuple(c) for c in self.prices.columns]

    @property
    def columns(self) -> List[Tuple[str, str]]:
        return [tuple(c) for c in self.merged.columns]

    def describe(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "md5_first_8mb": file_md5(Path(self.path)),
            "trading_rows": int(len(self.merged)),
            "observation_columns": int(self.merged.shape[1]),
            "indicator_columns": int(self.indicators.shape[1]),
            "price_columns": int(self.prices.shape[1]),
            "first_date": str(self.merged.index.min().date()),
            "last_date": str(self.merged.index.max().date()),
        }


def file_md5(path: Path, limit: int = 8_000_000) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def load_market_data(path: str | Path) -> MarketData:
    path = Path(path)
    prices = pd.read_hdf(path, "instruments")
    indicators = pd.read_hdf(path, "technical_indicators")
    prices.index = pd.to_datetime(prices.index)
    indicators.index = pd.to_datetime(indicators.index)
    merged = indicators.merge(prices, left_index=True, right_index=True).dropna()
    return MarketData(
        path=str(path), prices=prices, indicators=indicators, merged=merged.sort_index()
    )


def identical_data_files(paths: Sequence[Optional[str]]) -> pd.DataFrame:
    """Fingerprint every distinct data file referenced by the manifest."""
    rows = []
    for p in sorted({str(x) for x in paths if x}):
        path = Path(p)
        if not path.exists():
            rows.append({"path": p, "exists": False, "md5_first_8mb": None, "size_bytes": None})
            continue
        rows.append(
            {
                "path": p,
                "exists": True,
                "md5_first_8mb": file_md5(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)
