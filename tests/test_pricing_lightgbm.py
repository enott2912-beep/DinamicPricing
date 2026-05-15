"""Smoke- и контракт-тесты для LightGBM (experimental-режим)."""

import numpy as np
import pandas as pd
import pytest

from model.pricing import (
    LGBMRegressor,
    _build_lgbm_training_frame,
    _lgbm_data_warnings,
    fit_lightgbm_sales_model,
    predict_sales_lightgbm_with_pack,
    recommend_price_lightgbm,
    recommend_price_lightgbm_with_pack,
)

pytestmark = pytest.mark.skipif(LGBMRegressor is None, reason="lightgbm не установлен")


def test_lgbm_data_warnings_short_history(minimal_sales_df: pd.DataFrame):
    prep = _build_lgbm_training_frame(minimal_sales_df)
    warnings = _lgbm_data_warnings(prep)
    assert len(warnings) >= 1
    assert any("60" in w or "Наблюдений" in w for w in warnings)


def test_build_lgbm_training_frame_filters_oos(lgbm_training_df: pd.DataFrame):
    df = lgbm_training_df.copy()
    df.loc[df.index[-5:], "is_oos"] = True
    df.loc[df.index[-5:], "sales"] = 0.0
    prep = _build_lgbm_training_frame(df)
    assert not prep["is_oos"].any() if "is_oos" in prep.columns else True
    assert (prep["sales"] > 0).all()


def test_fit_lightgbm_short_history_unreliable(minimal_sales_df: pd.DataFrame):
    pack = fit_lightgbm_sales_model(minimal_sales_df)
    assert pack.model is None
    assert pack.reliable is False
    assert len(pack.warnings) > 0


def test_fit_lightgbm_empty_after_clean():
    df = pd.DataFrame(
        columns=["date", "our_price", "competitor_price", "sales", "is_oos"]
    )
    pack = fit_lightgbm_sales_model(df)
    assert pack.model is None
    assert pack.reliable is False


def test_fit_lightgbm_reliable_on_rich_history(lgbm_training_df: pd.DataFrame):
    pack = fit_lightgbm_sales_model(lgbm_training_df)
    assert pack.model is not None
    assert pack.reliable is True
    assert len(pack.features) > 0
    assert pack.p_min <= pack.p_max


def test_recommend_price_lightgbm_with_pack_respects_step_limit(lgbm_training_df: pd.DataFrame):
    pack = fit_lightgbm_sales_model(lgbm_training_df)
    last_price = 100.0
    result = recommend_price_lightgbm_with_pack(
        model_pack=pack,
        history_df=lgbm_training_df.tail(14),
        next_date=pd.Timestamp("2024-08-10"),
        last_price=last_price,
        competitor_price=102.0,
        cogs=60.0,
        max_daily_price_change_pct=2.0,
    )
    step = 0.02
    assert last_price * (1.0 - step) - 0.01 <= result["recommended_price"] <= last_price * (1.0 + step) + 0.01
    assert result["pred_sales"] >= 0.0


def test_recommend_price_lightgbm_fallback_when_unreliable(minimal_sales_df: pd.DataFrame):
    result = recommend_price_lightgbm(
        history_df=minimal_sales_df,
        next_date=pd.Timestamp("2025-01-10"),
        last_price=88.0,
        competitor_price=90.0,
        cogs=60.0,
    )
    assert result["reliable"] is False
    assert result["recommended_price"] == 88.0


def test_predict_sales_lightgbm_non_negative(lgbm_training_df: pd.DataFrame):
    pack = fit_lightgbm_sales_model(lgbm_training_df)
    pred = predict_sales_lightgbm_with_pack(
        model_pack=pack,
        history_df=lgbm_training_df.tail(14),
        next_date=pd.Timestamp("2024-08-10"),
        price=90.0,
        competitor_price=92.0,
    )
    assert pred >= 0.0


def test_predict_sales_fallback_when_no_model(minimal_sales_df: pd.DataFrame):
    pack = fit_lightgbm_sales_model(minimal_sales_df)
    expected = float(minimal_sales_df["sales"].tail(7).mean())
    pred = predict_sales_lightgbm_with_pack(
        model_pack=pack,
        history_df=minimal_sales_df,
        next_date=pd.Timestamp("2025-01-10"),
        price=90.0,
        competitor_price=92.0,
    )
    assert pred == expected
