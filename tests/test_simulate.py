"""Тесты цикла simulate (история → N дней вперёд)."""

from unittest.mock import patch

import pandas as pd
import pytest

from model.pricing import LGBMRegressor, simulate


def _as_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def _future_rows(result: pd.DataFrame, history_max: pd.Timestamp) -> pd.DataFrame:
    r = _as_dates(result)
    return r[r["date"] > history_max].copy()


def _n_entities(history: pd.DataFrame) -> int:
    cols = [c for c in ["store_id", "store", "brand_id", "brand", "product_id", "product"] if c in history.columns]
    if not cols:
        cols = ["product"]
    return history[cols].drop_duplicates().shape[0]


def test_simulate_appends_n_steps_days(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    n_steps = 5
    before = hist["date"].nunique()
    result = simulate(hist, n_steps=n_steps, method="rules")
    after = _as_dates(result)["date"].nunique()
    assert after == before + n_steps


def test_simulate_future_dates_after_history_max(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    max_hist = hist["date"].max()
    result = simulate(hist, n_steps=3, method="rules")
    fut = _future_rows(result, max_hist)
    assert len(fut) > 0
    assert (fut["date"] > max_hist).all()


def test_simulate_output_schema(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    result = simulate(hist, n_steps=2, method="rules")
    fut = _future_rows(result, hist["date"].max())
    for col in (
        "our_price",
        "sales",
        "revenue",
        "profit",
        "competitor_price",
        "competitor_1_price",
        "competitor_2_price",
        "is_oos",
    ):
        assert col in fut.columns
    for col in ("store", "brand", "product"):
        assert col in fut.columns


def test_simulate_invariants(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    result = simulate(hist, n_steps=5, method="regression", train_window_days=14)
    fut = _future_rows(result, hist["date"].max())
    assert (fut["our_price"] >= 1.0).all()
    assert (fut["sales"] >= 0).all()
    assert (fut["revenue"] >= 0).all()


def test_simulate_accounting(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    result = simulate(hist, n_steps=4, method="rules")
    fut = _future_rows(result, hist["date"].max())
    rev_ok = (fut["revenue"] - fut["sales"] * fut["our_price"]).abs() <= 0.01
    prof_ok = (fut["profit"] - (fut["revenue"] - fut["sales"] * fut["cogs"])).abs() <= 0.01
    assert rev_ok.all()
    assert prof_ok.all()


def test_simulate_reproducible_rules(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    kwargs = dict(n_steps=4, method="rules")
    a = _future_rows(simulate(hist, **kwargs), hist["date"].max())
    b = _future_rows(simulate(hist, **kwargs), hist["date"].max())
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_dtype=False,
    )


def test_simulate_empty_or_no_products():
    empty = pd.DataFrame(columns=["date", "product", "our_price", "sales", "revenue", "cogs", "profit"])
    assert simulate(empty, n_steps=5, method="rules").empty

    unknown = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "product": "НеизвестныйТовар",
                "our_price": 10.0,
                "sales": 1.0,
                "revenue": 10.0,
                "cogs": 5.0,
                "profit": 5.0,
            }
        ]
    )
    out = simulate(_as_dates(unknown), n_steps=3, method="rules")
    assert out["date"].nunique() == 1


def test_simulate_target_product_filter(sim_history_multi_entity: pd.DataFrame):
    hist = _as_dates(sim_history_multi_entity)
    result = simulate(hist, n_steps=3, method="rules", target_product="Молоко")
    fut = _future_rows(result, hist["date"].max())
    assert set(fut["product"].unique()) == {"Молоко"}


def test_simulate_row_count(sim_history_multi_entity: pd.DataFrame):
    hist = _as_dates(sim_history_multi_entity)
    n_steps = 4
    entities = _n_entities(hist)
    result = simulate(hist, n_steps=n_steps, method="rules")
    fut = _future_rows(result, hist["date"].max())
    assert len(fut) == n_steps * entities


def test_simulate_regression_completes(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    result = simulate(
        hist,
        n_steps=5,
        method="regression",
        train_window_days=14,
        retrain_every_days=2,
    )
    fut = _future_rows(result, hist["date"].max())
    assert len(fut) == 5


def test_simulate_regression_daily_price_step_limit(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    pct = 2.0
    result = simulate(
        hist,
        n_steps=6,
        method="regression",
        train_window_days=14,
        max_daily_price_change_pct=pct,
    )
    fut = _future_rows(result, hist["date"].max())
    step = pct / 100.0
    eps = 0.05
    for _, grp in fut.groupby("product", sort=False):
        prices = grp.sort_values("date")["our_price"].to_numpy(dtype=float)
        last_hist = float(
            hist[(hist["product"] == grp["product"].iloc[0])].sort_values("date")["our_price"].iloc[-1]
        )
        prev = last_hist
        for p in prices:
            assert p >= prev * (1.0 - step) - eps
            assert p <= prev * (1.0 + step) + eps
            prev = p


def test_simulate_regression_train_window_clipped(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    result = simulate(
        hist,
        n_steps=2,
        method="regression",
        train_window_days=999,
    )
    assert len(_future_rows(result, hist["date"].max())) == 2


def test_simulate_rules_completes(sim_history_df: pd.DataFrame):
    hist = _as_dates(sim_history_df)
    result = simulate(hist, n_steps=5, method="rules")
    assert len(_future_rows(result, hist["date"].max())) == 5


def test_simulate_rules_with_patched_engine(sim_history_df: pd.DataFrame, rule_engine):
    hist = _as_dates(sim_history_df)

    class _FixedEngine:
        rules = [{"name": "up", "condition": "price > 0", "action": "price * 1.10"}]

        def evaluate(self, context):
            return round(float(context["price"]) * 1.10, 2), "up"

    with patch("model.pricing.get_rule_engine", return_value=_FixedEngine()):
        result = simulate(hist, n_steps=1, method="rules")
    fut = _future_rows(result, hist["date"].max())
    last_hist_price = float(hist.sort_values("date")["our_price"].iloc[-1])
    assert fut["our_price"].iloc[0] == pytest.approx(round(last_hist_price * 1.10, 2), rel=0, abs=0.02)


@pytest.mark.skipif(LGBMRegressor is None, reason="lightgbm не установлен")
def test_simulate_lightgbm_completes(lgbm_training_df: pd.DataFrame):
    hist = _as_dates(lgbm_training_df)
    result = simulate(
        hist,
        n_steps=2,
        method="lightgbm",
        train_window_days=60,
        retrain_every_days=7,
    )
    fut = _future_rows(result, hist["date"].max())
    assert len(fut) == 2
    assert (fut["sales"] >= 0).all()


@pytest.mark.skipif(LGBMRegressor is None, reason="lightgbm не установлен")
def test_simulate_lightgbm_short_history(minimal_sales_df: pd.DataFrame):
    hist = _as_dates(minimal_sales_df)
    result = simulate(hist, n_steps=2, method="lightgbm", train_window_days=7)
    fut = _future_rows(result, hist["date"].max())
    assert len(fut) == 2
    assert (fut["sales"] >= 0).all()
