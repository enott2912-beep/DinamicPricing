"""
Тест-страховка: порог вариативности цены в LightGBM-проверке (_lgbm_data_warnings)
должен быть МАСШТАБНО-ИНВАРИАНТНЫМ (коэффициент вариации), а не абсолютным.

Контекст: старый порог LGBM_MIN_PRICE_STD=1.0 сравнивал std цены напрямую с
константой 1.0 — это работает для цен в рублях (80-500 руб), но ошибочно
браковал данные с маленькой абсолютной ценой и заметной ОТНОСИТЕЛЬНОЙ
вариативностью. Обнаружено на внешнем датасете Avocado Prices (Kaggle):
цена ~$1.36, std=$0.21 (CV≈15%) — реальная, существенная вариативность,
но абсолютный порог (std=0.21 < 1.0) отбрасывал ВСЕ 108 сущностей датасета
ещё до этапа holdout-валидации качества модели, не позволяя её вообще
проверить на данных с такой шкалой цены.

Источник фикстуры: датасет Avocado Prices (Kaggle, открытый, через зеркало
github.com/synle/machine-learning-sample-dataset), обрезан до 3 регионов
(Albany, Atlanta, Boston) x 2 типов, чтобы не раздувать репозиторий.
"""
from pathlib import Path

import pandas as pd
import pytest

from model.pricing import LGBM_MIN_PRICE_CV, _lgbm_data_warnings, _build_lgbm_training_frame, fit_lightgbm_sales_model

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "external_avocado_weekly_sample.csv"


@pytest.fixture(scope="module")
def avocado_df() -> pd.DataFrame:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Файл с датасетом Avocado не найден: {FIXTURE_PATH}")
    df = pd.read_csv(FIXTURE_PATH)
    df["date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["region", "type", "date"])
    weekly_avg = df.groupby(["date", "type"])["AveragePrice"].mean().rename("competitor_price")
    df = df.merge(weekly_avg, on=["date", "type"])
    return df


def _entity_frame(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "date": g["date"],
        "our_price": g["AveragePrice"],
        "competitor_price": g["competitor_price"],
        "sales": g["Total Volume"] / 1000.0,
        "is_oos": False,
    })


def test_avocado_fixture_has_low_absolute_but_real_relative_variability(avocado_df: pd.DataFrame):
    """
    Sanity-check самой фикстуры: абсолютный std цены маленький (<1.0 — старый
    жёсткий порог), но коэффициент вариации заметно выше нового порога.
    Если это не так, тест ниже ничего не проверяет.
    """
    g = avocado_df[(avocado_df["region"] == "Albany") & (avocado_df["type"] == "conventional")]
    price = g["AveragePrice"]
    cv = float(price.std(ddof=0) / price.mean())
    assert price.std(ddof=0) < 1.0, "Фикстура должна иметь маленький абсолютный std цены для этого теста"
    assert cv > LGBM_MIN_PRICE_CV, f"CV={cv:.2%} должен быть выше порога {LGBM_MIN_PRICE_CV:.0%}, иначе кейс не воспроизводится"


def test_lgbm_price_variability_threshold_is_scale_invariant(avocado_df: pd.DataFrame):
    """
    Регрессионный барьер: датасет с маленькой абсолютной ценой, но реальной
    относительной вариативностью (~15% CV), не должен получать предупреждение
    "Слишком узкий диапазон цен" только из-за масштаба валюты/единиц измерения.
    """
    rejected_purely_on_price_warning = 0
    total = 0
    for (region, atype), g in avocado_df.groupby(["region", "type"]):
        total += 1
        entity_df = _entity_frame(g.sort_values("date"))
        prep = _build_lgbm_training_frame(entity_df)
        warnings = _lgbm_data_warnings(prep)
        price_warnings = [w for w in warnings if "диапазон цен" in w]
        if price_warnings:
            rejected_purely_on_price_warning += 1

    assert rejected_purely_on_price_warning == 0, (
        f"{rejected_purely_on_price_warning}/{total} сущностей отброшены из-за предупреждения "
        "о узком диапазоне цен, хотя их коэффициент вариации реально высокий. "
        "Проверьте, не вернулся ли абсолютный порог LGBM_MIN_PRICE_STD вместо LGBM_MIN_PRICE_CV "
        "в model/pricing.py: _lgbm_data_warnings."
    )


def test_lgbm_fits_and_runs_holdout_validation_on_avocado_data(avocado_df: pd.DataFrame):
    """
    Сквозной тест: на датасете с маленькой ценой (что раньше отсекалось на
    этапе data_warnings) holdout-проверка качества (R²/MAE) должна реально
    запускаться и отдавать числовые значения, а не падать в None.
    """
    g = avocado_df[(avocado_df["region"] == "Albany") & (avocado_df["type"] == "conventional")].sort_values("date")
    entity_df = _entity_frame(g)
    pack = fit_lightgbm_sales_model(entity_df, tune=False)

    assert pack.model is not None, f"Модель не обучилась, warnings: {pack.warnings}"
    assert pack.r2_holdout is not None, "Holdout-валидация должна была запуститься и вернуть R²"
    assert pack.mae_ratio is not None
