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
    """
    ≥60 дней, вариативные цены — достаточно для обучения LightGBM после лагов.

    Цена колеблется синусоидой БЕЗ линейного тренда: если бы цена со временем
    дрейфовала вверх/вниз (как было раньше: + np.linspace(-4, 4, n)), отложенный
    отрезок (последние 20% по времени, holdout) видел бы цены за пределами
    диапазона, на котором обучалась модель — деревья LightGBM не умеют
    экстраполировать, и holdout R^2 уходил в сильный минус не из-за плохой
    модели, а из-за самой фикстуры. См. model/pricing.py: fit_lightgbm_sales_model
    (holdout-валидация по времени, добавлена для проверки качества после фита).
    """
    n = 70
    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    prices = 85.0 + 8.0 * np.sin(np.linspace(0, 5 * np.pi, n))
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


def _make_sim_row(
    date: pd.Timestamp,
    product: str,
    product_id: int,
    our_price: float,
    *,
    store_id: int = 1,
    store: str = "TestStore",
    store_profile: str = "У дома",
    brand_id: int = 1,
    brand: str = "TestBrand",
    cogs: float = 60.0,
) -> dict:
    s = float(max(5.0, round(420.0 - 2.8 * our_price)))
    comp = round(our_price * 1.02, 2)
    rev = round(s * our_price, 2)
    return {
        "date": date,
        "store_id": store_id,
        "store": store,
        "store_profile": store_profile,
        "brand_id": brand_id,
        "brand": brand,
        "product_id": product_id,
        "product": product,
        "our_price": float(our_price),
        "competitor_1_price": round(comp * 1.01, 2),
        "competitor_2_price": round(comp * 0.99, 2),
        "competitor_price": comp,
        "sales": s,
        "revenue": rev,
        "cogs": cogs,
        "profit": round(rev - s * cogs, 2),
        "is_oos": False,
    }


@pytest.fixture
def sim_history_df() -> pd.DataFrame:
    """21 день, одна сущность (магазин × бренд × Молоко) — для simulate."""
    dates = pd.date_range("2025-03-01", periods=21, freq="D")
    prices = np.linspace(78, 98, len(dates))
    rows = [_make_sim_row(d, "Молоко", 1, float(round(p, 2))) for d, p in zip(dates, prices)]
    return pd.DataFrame(rows)


@pytest.fixture
def sim_history_multi_entity() -> pd.DataFrame:
    """Две SKU в одном магазине — для проверки фильтра и числа строк."""
    dates = pd.date_range("2025-04-01", periods=14, freq="D")
    rows = []
    for d in dates:
        rows.append(_make_sim_row(d, "Молоко", 1, 85.0))
        rows.append(_make_sim_row(d, "Кофе", 4, 450.0, cogs=280.0))
    return pd.DataFrame(rows)
