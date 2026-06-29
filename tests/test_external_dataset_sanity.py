"""
Тест-страховка на ВНЕШНЕМ датасете (не на синтетике из generator/).

Источник: открытый датасет "Retail Price Optimization" (Kaggle / зеркало на
GitHub: fidanmammadova03/price-optimization-python, файл retail_price.csv).
Файл лежит в tests/fixtures/external_retail_price_sample.csv.

Зачем этот тест существует:
  При разработке выяснилось, что критерий "надёжности" линейной регрессии
  (model/pricing.py: _fit_linear_sales_vs_price), проверявший только знак
  коэффициента цена->спрос, помечал ~65% товаров из внешнего датасета как
  "надёжные", хотя на отложенном отрезке (out-of-sample) их R^2 был сильно
  отрицательным (модель хуже константы). После добавления порога по R^2
  доля "надёжных" упала до ~19%, и она лучше коррелирует с реальным
  качеством прогноза (см. история коммита / TESTS.md).

  Этот тест не проверяет "хорошее" абсолютное качество модели (на таких
  скудных данных — 5-20 точек на товар, месячная агрегация — это
  объективно недостижимо), а служит регрессионным барьером: если кто-то
  в будущем случайно ослабит критерий надёжности обратно до "только знак
  коэффициента", этот тест должен покраснеть.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.pricing import (
    MIN_R2_FOR_RELIABLE_FIT,
    _fit_linear_sales_vs_price,
    _ols_fit_diagnostics,
    _confidence_weight,
    simulate,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "external_retail_price_sample.csv"


@pytest.fixture(scope="module")
def external_df() -> pd.DataFrame:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Файл с внешним датасетом не найден: {FIXTURE_PATH}")
    df = pd.read_csv(FIXTURE_PATH)
    df["date"] = pd.to_datetime(df["month_year"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["product_id", "date"])
    return df


def test_external_dataset_loads_with_expected_columns(external_df: pd.DataFrame):
    expected_cols = {"product_id", "unit_price", "qty", "comp_1", "freight_price"}
    assert expected_cols.issubset(set(external_df.columns))
    assert external_df["product_id"].nunique() >= 30
    assert len(external_df) >= 300


def test_reliability_rate_is_not_inflated(external_df: pd.DataFrame):
    """
    Регрессионный барьер: доля "надёжных" по новому критерию должна оставаться
    заметно ниже, чем по старому критерию "только знак коэффициента" (~65%).
    Если доля снова подскочит к этому уровню — кто-то ослабил пороги.
    """
    reliable_flags = []
    for _, g in external_df.groupby("product_id"):
        prices = g["unit_price"].values.astype(float)
        sales = g["qty"].values.astype(float)
        cogs_proxy = float(g["freight_price"].mean())
        _, _, _, reliable = _fit_linear_sales_vs_price(
            prices, sales, fallback_price=float(prices.mean()), cogs=cogs_proxy
        )
        reliable_flags.append(reliable)

    reliable_rate = float(np.mean(reliable_flags))
    assert reliable_rate < 0.40, (
        f"Доля 'надёжных' SKU = {reliable_rate:.1%} — это близко к значению старого, "
        "слабого критерия (только знак коэффициента, ~65%). Проверьте, не ослаблен "
        "ли порог MIN_R2_FOR_RELIABLE_FIT в model/pricing.py."
    )


def test_confidence_weight_is_fractional_for_weak_fits(external_df: pd.DataFrame):
    """
    Для слабого, но формально отрицательного коэффициента вес доверия должен
    быть строго между 0 и 1 (а не бинарным 0/1, как было раньше) — это и есть
    смешивание (blending), а не жёсткое переключение.
    """
    found_fractional_weight = False
    for _, g in external_df.groupby("product_id"):
        prices = g["unit_price"].values.astype(float)
        sales = g["qty"].values.astype(float)
        diag = _ols_fit_diagnostics(prices, sales)
        if diag["n"] < 5:
            continue
        weight = _confidence_weight(diag["r2"], diag["coef"])
        assert 0.0 <= weight <= 1.0
        if diag["coef"] < 0 and 0.0 < diag["r2"] < MIN_R2_FOR_RELIABLE_FIT:
            assert 0.0 < weight < 1.0
            found_fractional_weight = True

    assert found_fractional_weight, "В датасете должен быть хотя бы один товар со слабым, но не нулевым фитом"


def test_simulate_regression_runs_on_external_data_without_crashing(external_df: pd.DataFrame):
    """
    Сквозной дымовой тест: вся симуляция (simulate(method='regression')) должна
    отработать на внешних, шумных, разреженных данных без исключений и выдать
    конечные положительные цены — это главный риск при работе с данными,
    непохожими на собственный синтетический генератор.
    """
    sim_in = pd.DataFrame({
        "date": external_df["date"],
        "product": external_df["product_id"],
        "our_price": external_df["unit_price"],
        "competitor_price": external_df["comp_1"],
        "sales": external_df["qty"],
        "is_oos": False,
        "cogs": external_df["freight_price"],
    })

    result = simulate(
        sim_in,
        n_steps=3,
        method="regression",
        retrain_every_days=1,
        train_window_days=999,
        max_daily_price_change_pct=8.0,
    )

    future = result[result["date"] > sim_in["date"].max()]
    assert len(future) > 0
    assert np.isfinite(future["our_price"]).all()
    assert (future["our_price"] > 0).all()
    assert np.isfinite(future["sales"]).all()
    assert (future["sales"] >= 0).all()
