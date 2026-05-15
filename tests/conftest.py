from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.rules_engine import RuleEngine


@pytest.fixture
def minimal_sales_df() -> pd.DataFrame:
    """Минимально валидный датасет одного SKU для регрессии и forecast."""
    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    prices = np.linspace(80, 95, len(dates))
    sales = np.maximum(0, np.round(320 - 2.5 * prices))
    rows = []
    for d, p, s in zip(dates, prices, sales):
        rev = round(s * p, 2)
        cogs = 60.0
        rows.append(
            {
                "date": d,
                "product_id": 1,
                "product": "Молоко",
                "our_price": float(p),
                "competitor_price": float(p) * 1.02,
                "sales": float(s),
                "revenue": rev,
                "cogs": cogs,
                "profit": round(rev - s * cogs, 2),
                "is_oos": False,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_rules() -> list[dict]:
    return [
        {
            "name": "always_match",
            "condition": "price > 0",
            "action": "price * 1.05",
        },
        {
            "name": "never_reached",
            "condition": "price < 0",
            "action": "1.0",
        },
    ]


@pytest.fixture
def rule_engine(tmp_path: Path, sample_rules: list[dict]) -> RuleEngine:
    rules_path = tmp_path / "rules.json"
    engine = RuleEngine(rules_path=rules_path)
    engine.rules = sample_rules
    return engine
