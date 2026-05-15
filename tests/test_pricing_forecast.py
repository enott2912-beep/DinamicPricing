from unittest.mock import patch

import pandas as pd

from model.pricing import PRODUCTS, apply_rules, forecast


def test_forecast_with_regression_params():
    result = forecast("Молоко", recommended_price=90.0, current_metric=1000.0, regression_params=(400.0, 3.0))
    assert result["forecast_sales"] == max(0, round(400.0 - 3.0 * 90.0))
    assert result["forecast_profit"] == round(result["forecast_sales"] * (90.0 - PRODUCTS["Молоко"]["cogs"]), 2)
    assert result["growth_pct"] == round((result["forecast_profit"] - 1000.0) / 1000.0 * 100, 1)


def test_forecast_without_regression_uses_products():
    p = PRODUCTS["Молоко"]
    price = p["base_price"] + 10.0
    result = forecast("Молоко", recommended_price=price, current_metric=500.0, regression_params=None)
    expected_sales = max(0, round(p["base_sales"] - p["elasticity"] * 10.0))
    assert result["forecast_sales"] == expected_sales


def test_apply_rules_delegates_to_engine(rule_engine):
    row = pd.Series(
        {
            "our_price": 100.0,
            "competitor_1_price": 105.0,
            "competitor_2_price": 95.0,
            "competitor_price": 95.0,
            "sales": 200.0,
            "cogs": 60.0,
        }
    )
    with patch("model.pricing.get_rule_engine", return_value=rule_engine):
        price, name = apply_rules(row, avg_sales_7d=210.0)
    assert name == "always_match"
    assert price == 105.0
