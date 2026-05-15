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


@pytest.fixture
def lgbm_training_df() -> pd.DataFrame:
    """≥60 дней, вариативные цены — достаточно для обучения LightGBM после лагов."""
    n = 70
    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    prices = 85.0 + 8.0 * np.sin(np.linspace(0, 5 * np.pi, n)) + np.linspace(-4, 4, n)
    rows = []
    for d, p in zip(dates, prices):
        p = float(round(p, 2))
        s = float(max(5.0, round(420.0 - 2.8 * p)))
        comp = round(p * 1.02, 2)
        rev = round(s * p, 2)
        cogs = 60.0
        rows.append(
            {
                "date": d,
                "product_id": 1,
                "product": "Молоко",
                "our_price": p,
                "competitor_price": comp,
                "sales": s,
                "revenue": rev,
                "cogs": cogs,
                "profit": round(rev - s * cogs, 2),
                "is_oos": False,
            }
        )
    return pd.DataFrame(rows)
