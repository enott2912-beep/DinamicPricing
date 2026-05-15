import numpy as np
import pandas as pd

from model.pricing import (
    PRODUCTS,
    _fit_linear_sales_vs_price,
    fit_regression,
    predict_sales_regression,
    products_in_dataframe,
)


def test_fit_linear_sales_vs_price_insufficient_data():
    prices = np.array([80.0, 85.0])
    sales = np.array([200.0, 190.0])
    a, b, opt, reliable = _fit_linear_sales_vs_price(prices, sales, fallback_price=80.0, cogs=60.0)
    assert reliable is False
    assert a == 0.0
    assert b == 0.0
    assert opt == 80.0


def test_fit_linear_sales_vs_price_negative_slope_reliable():
    prices = np.array([70.0, 80.0, 90.0, 100.0, 110.0])
    sales = 400.0 - 3.0 * prices
    a, b, opt, reliable = _fit_linear_sales_vs_price(prices, sales, fallback_price=85.0, cogs=60.0)
    assert reliable is True
    assert b > 0
    assert opt >= 1.0


def test_fit_linear_optimal_price_near_formula():
    prices = np.linspace(70, 110, 12)
    sales = 500.0 - 3.5 * prices
    a, b, opt, reliable = _fit_linear_sales_vs_price(prices, sales, fallback_price=85.0, cogs=60.0)
    assert reliable is True
    expected_raw = (a + b * 60.0) / (2.0 * b)
    assert abs(opt - expected_raw) < 15.0


def test_fit_regression_filters_oos(minimal_sales_df: pd.DataFrame):
    df = minimal_sales_df.copy()
    df.loc[df.index[-3:], "is_oos"] = True
    df.loc[df.index[-3:], "sales"] = 999.0
    a, b, opt, reliable = fit_regression(df, "Молоко")
    assert reliable is True
    assert b > 0


def test_fit_regression_filters_zero_sales(minimal_sales_df: pd.DataFrame):
    df = minimal_sales_df.copy()
    df.loc[df.index[0], "sales"] = 0.0
    a, b, _, reliable = fit_regression(df, "Молоко")
    assert reliable is True


def test_predict_sales_regression_formula():
    assert predict_sales_regression(400.0, 3.0, 90.0, noise=0.0) == max(0, int(round(400 - 3 * 90)))


def test_predict_sales_regression_with_noise():
    assert predict_sales_regression(400.0, 3.0, 90.0, noise=2.5) == max(0, int(round(400 - 3 * 90 + 2.5)))


def test_products_in_dataframe_order(minimal_sales_df: pd.DataFrame):
    df = pd.concat(
        [
            minimal_sales_df,
            minimal_sales_df.assign(product="Кофе", product_id=4),
        ],
        ignore_index=True,
    )
    result = products_in_dataframe(df)
    product_keys = list(PRODUCTS.keys())
    assert result == [p for p in product_keys if p in {"Молоко", "Кофе"}]
