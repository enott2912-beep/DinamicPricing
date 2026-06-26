"""Загруженный CSV: произвольные SKU в рекомендациях и симуляции."""

import pandas as pd

from model.analytics import get_recommendations_all_products
from model.pricing import forecast, infer_entity_params, simulate


def test_simulate_custom_product_no_keyerror(minimal_sales_df: pd.DataFrame):
    df = minimal_sales_df.assign(product="Яблоко", product_id=99)
    result = simulate(df, n_steps=3, method="rules", target_product="Яблоко")
    assert len(result) > len(df)
    future = result[result["date"] > df["date"].max()]
    assert (future["product"] == "Яблоко").all()


def test_recommendations_custom_products(minimal_sales_df: pd.DataFrame):
    df = minimal_sales_df.assign(product="Яблоко", product_id=99)
    res = get_recommendations_all_products(df, df, include_lightgbm=False)
    assert res["rule_rows"]
    assert res["rule_rows"][0]["Товар"] == "Яблоко"


def test_forecast_with_inferred_params(minimal_sales_df: pd.DataFrame):
    params = infer_entity_params(minimal_sales_df, "Яблоко")
    df = minimal_sales_df.assign(product="Яблоко")
    last_price = float(df["our_price"].iloc[-1])
    result = forecast("Яблоко", last_price, 100.0, product_params=params)
    assert result["forecast_sales"] >= 0
    assert result["forecast_profit"] >= 0
