"""Feature taxonomy for attribution diagnostics.

Reuses the manuscript-consistent indicator parsing convention from the audit
compute core (instrument_indicator_lag) and extends it with deterministic
family regexes covering price, returns, momentum, volatility, trend, volume,
oscillators, temporal variables, RTG, rewards, actions, and CASH.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Ordered rules: first match wins.
_FEATURE_RULES: List[Tuple[str, str, re.Pattern[str]]] = [
    ("return_to_go", "regex:rtg|return_to_go|returns_to_go", re.compile(r"(rtg|return[_-]?to[_-]?go|returns_to_go)", re.I)),
    ("rewards", "regex:reward", re.compile(r"(^|_)reward(s)?($|_)", re.I)),
    ("actions", "regex:action|allocation|weight", re.compile(r"(^|_)(action|allocation|portfolio_weight)s?($|_)", re.I)),
    ("cash", "regex:cash", re.compile(r"(^|_)cash($|_)", re.I)),
    ("temporal", "regex:timestep|time_idx|calendar|weekday|month", re.compile(r"(timestep|time[_-]?idx|calendar|weekday|month|dayofweek)", re.I)),
    ("volume", "regex:volume|obv|ad_line|mfi|chaikin", re.compile(r"(volume|obv|adline|ad_line|mfi|chaikin|ADOSC|AD\b)", re.I)),
    ("volatility", "regex:atr|natr|std|volatility|vix|bbwidth|true_range", re.compile(r"(atr|natr|volatility|std|vix|bbwidth|true[_-]?range|TRANGE)", re.I)),
    ("trend", "regex:sma|ema|wma|dema|tema|trix|adx|aroon|psar|supertrend", re.compile(r"(sma|ema|wma|dema|tema|trix|adx|aroon|psar|supertrend|MACD)", re.I)),
    ("momentum", "regex:rsi|roc|mom|cmo|willr|stoch|cci|ultosc", re.compile(r"(rsi|roc|mom|momentum|cmo|willr|stoch|cci|ultosc|apo|ppo)", re.I)),
    ("oscillators", "regex:osc|bop|ht_", re.compile(r"(osc|bop|ht_|fisher|kst)", re.I)),
    ("raw_returns", "regex:return|pct_change|logret", re.compile(r"(^|_)(ret|return|pct[_-]?change|logret|log_return)s?($|_)", re.I)),
    ("price_level", "regex:open|high|low|close|price|adj_close", re.compile(r"(open|high|low|close|adj[_-]?close|price)", re.I)),
    ("instrument_returns", "regex:instrument_return", re.compile(r"instrument[_-]?return", re.I)),
    ("technical_indicators", "regex:ti_|tech_", re.compile(r"(^|_)(ti|tech(nical)?[_-]?ind)", re.I)),
]


@dataclass(frozen=True)
class FeatureMapping:
    feature_name: str
    feature_family: str
    mapping_rule: str
    mapping_status: str


def strip_lag_suffix(name: str) -> str:
    text = str(name)
    text = re.sub(r"_lb_\d+$", "", text, flags=re.I)
    text = re.sub(r"_\d+$", "", text)
    return text


def parse_indicator_token(name: str) -> str:
    """Extract indicator token from instrument_indicator[_lag] labels."""
    text = str(name)
    base = strip_lag_suffix(text)
    parts = base.split("_")
    if len(parts) >= 2:
        # Drop leading instrument ticker-like token(s).
        if parts[0].startswith("^") or parts[0].isupper() or parts[0] in {"CASH"}:
            return "_".join(parts[1:])
    return base


def map_feature_name(name: str) -> FeatureMapping:
    text = str(name)
    probe = f"{text} {strip_lag_suffix(text)} {parse_indicator_token(text)}"
    for family, rule, pattern in _FEATURE_RULES:
        if pattern.search(probe):
            return FeatureMapping(
                feature_name=text,
                feature_family=family,
                mapping_rule=rule,
                mapping_status="mapped",
            )
    return FeatureMapping(
        feature_name=text,
        feature_family="unclassified",
        mapping_rule="fallback:unclassified",
        mapping_status="unclassified",
    )


def build_taxonomy(feature_names: Iterable[str]) -> pd.DataFrame:
    rows = [map_feature_name(name).__dict__ for name in feature_names]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["feature_name", "feature_family", "mapping_rule", "mapping_status"]
        )
    return (
        df.drop_duplicates(subset=["feature_name"])
        .sort_values(["feature_family", "feature_name"])
        .reset_index(drop=True)
    )


def family_labels_for_names(feature_names: Sequence[str]) -> List[str]:
    return [map_feature_name(name).feature_family for name in feature_names]
