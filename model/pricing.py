import numpy as np
import pandas as pd
import logging
import zlib
from dataclasses import dataclass

from model.math_engine import (
    calc_competitor_1_prices,
    calc_competitor_2_prices,
    calc_demand_regression,
    calc_demand_rules,
)
from model.rules_engine import get_rule_engine
from model.utils import parse_is_oos_series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - fallback если пакет не установлен
    LGBMRegressor = None

# ############################################################################
# 1. КОНСТАНТЫ И КОНФИГУРАЦИЯ (Единый источник правды)
# ############################################################################

SEED = 42
logger = logging.getLogger(__name__)

# Базовая информация о товарах: ID, стартовая цена, эластичность, спрос, себестоимость
PRODUCTS = {
    'Молоко':   {'id': 1, 'base_price': 80,  'elasticity': 2.0, 'base_sales': 300, 'cogs': 60},
    'Хлеб':     {'id': 2, 'base_price': 50,  'elasticity': 1.5, 'base_sales': 250, 'cogs': 30},
    'Сок':      {'id': 3, 'base_price': 120, 'elasticity': 3.0, 'base_sales': 150, 'cogs': 80},
    'Кофе':     {'id': 4, 'base_price': 450, 'elasticity': 1.2, 'base_sales': 80,  'cogs': 280},
    'Шоколад':  {'id': 5, 'base_price': 100, 'elasticity': 2.5, 'base_sales': 200, 'cogs': 65},
}

SEASONALITY_PHASE = {
    'Кофе': -np.pi / 2,   # зимний пик
    'Сок': np.pi / 2,     # летний пик
    'Молоко': 0.0,
    'Хлеб': 0.0,
    'Шоколад': -np.pi / 3,
}

LGBM_MIN_ROWS = 60
LGBM_MIN_UNIQUE_PRICES = 8
# Порог вариативности цены задан коэффициентом вариации (std/mean), а не
# абсолютным std. Старый порог LGBM_MIN_PRICE_STD=1.0 был калиброван под
# рублёвые цены (80-500 руб) и ошибочно отбраковывал валидные данные с малой
# абсолютной ценой, но заметной относительной вариативностью — например,
# датасет Avocado Prices (Kaggle): цена ~$1.36, std=$0.21 (CV≈15%), что
# было полностью отброшено старым порогом (std=0.21 < 1.0), хотя 15% — это
# существенная вариативность для обучения модели. См. tests/test_lgbm_thresholds_scale_invariant.py.
LGBM_MIN_PRICE_CV = 0.03
DEFAULT_ELASTICITY = 2.0

# Пороги надёжности линейной регрессии Sales ≈ A - B*Price.
# Раньше проверялся только знак коэффициента — что пропускало модели
# с отрицательным R^2 на разреженных/шумных данных (см. валидацию на внешнем
# датасете Retail Price Optimization, Kaggle). Теперь дополнительно проверяем
# качество фита и минимальное число точек.
MIN_POINTS_FOR_RELIABLE_FIT = 5
MIN_R2_FOR_RELIABLE_FIT = 0.25

LGBM_R2_THRESHOLD = 0.20
LGBM_MAE_RATIO_MAX = 0.50
LGBM_HOLDOUT_RATIO = 0.20
LGBM_HOLDOUT_MIN_ROWS = 5
LGBM_PRICE_QUANTILE_LO = 0.05
LGBM_PRICE_QUANTILE_HI = 0.95
LGBM_EXTRAP_DECAY_WIDTH = 0.10


@dataclass
class LGBMModelPack:
    model: object | None
    features: list[str]
    reliable: bool
    warnings: list[str]
    p_min: float
    p_max: float
    trust_weight: float = 1.0
    r2_holdout: float | None = None
    mae_ratio: float | None = None


def _lgbm_data_warnings(df: pd.DataFrame) -> list[str]:
    msgs: list[str] = []
    n_rows = len(df)
    if n_rows < LGBM_MIN_ROWS:
        msgs.append(f"Наблюдений мало: {n_rows} (< {LGBM_MIN_ROWS}).")
    uniq_prices = int(df["our_price"].nunique()) if "our_price" in df.columns and n_rows > 0 else 0
    if uniq_prices < LGBM_MIN_UNIQUE_PRICES:
        msgs.append(f"Слабая вариативность цены: уникальных значений {uniq_prices} (< {LGBM_MIN_UNIQUE_PRICES}).")
    if n_rows > 1:
        price_mean = float(df["our_price"].mean())
        price_std = float(df["our_price"].std(ddof=0))
        price_cv = price_std / price_mean if price_mean > 0 else 0.0
        if price_cv < LGBM_MIN_PRICE_CV:
            msgs.append(f"Слишком узкий диапазон цен: CV={price_cv:.1%} (std={price_std:.2f}, mean={price_mean:.2f}).")
    return msgs


def _build_lgbm_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "is_oos" in work.columns:
        work = work[~parse_is_oos_series(work["is_oos"])]
    work = work[work["sales"] > 0].copy()
    if work.empty:
        return work
    work = work.sort_values("date").copy()
    work["date"] = pd.to_datetime(work["date"])
    day_of_year = work["date"].dt.dayofyear.astype(float)
    work["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
    work["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)
    work["dow"] = work["date"].dt.dayofweek.astype(int)
    work["month"] = work["date"].dt.month.astype(int)
    work["price_gap"] = work["our_price"] - work["competitor_price"]
    work["sales_lag1"] = work["sales"].shift(1)
    work["sales_lag7"] = work["sales"].shift(7)
    work["sales_roll7"] = work["sales"].shift(1).rolling(7, min_periods=2).mean()
    return work.dropna().copy()


LGBM_TUNE_MIN_ROWS = 90  # тюнинг имеет смысл только при заметном объёме данных
LGBM_TUNE_N_ITER = 4
LGBM_DEFAULT_PARAMS = dict(
    n_estimators=180,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
)


def _tune_lgbm_params(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Лёгкий перенос идеи из notebooks/01_model_training_and_evaluation.ipynb в прод:
    маленький RandomizedSearchCV с TimeSeriesSplit вместо захардкоженных параметров.
    Намеренно дёшево (n_iter=4) — вызывается на каждом переобучении в simulate().
    При любой ошибке или нехватке данных тихо откатывается на LGBM_DEFAULT_PARAMS,
    чтобы не ронять симуляцию.
    """
    if len(X) < LGBM_TUNE_MIN_ROWS:
        return LGBM_DEFAULT_PARAMS
    try:
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        from scipy.stats import randint, uniform

        dist = {
            "n_estimators": randint(60, 200),
            "num_leaves": randint(8, 50),
            "learning_rate": uniform(0.02, 0.13),
            "min_child_samples": randint(10, 30),
        }
        search = RandomizedSearchCV(
            LGBMRegressor(random_state=SEED, verbose=-1, subsample=0.85, colsample_bytree=0.85),
            dist,
            n_iter=LGBM_TUNE_N_ITER,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            random_state=SEED,
        ).fit(X, y)
        best = dict(search.best_params_)
        best.setdefault("subsample", 0.85)
        best.setdefault("colsample_bytree", 0.85)
        best.setdefault("max_depth", -1)
        return best
    except Exception as exc:  # подбор не должен ронять симуляцию
        logger.warning(f"LightGBM-тюнинг не удался, использую дефолтные параметры: {exc}")
        return LGBM_DEFAULT_PARAMS


def _lgbm_trust_weight(r2: float, mae_ratio: float) -> float:
    """Плавный вес доверия LightGBM по holdout-метрикам."""
    if r2 < LGBM_R2_THRESHOLD or mae_ratio > LGBM_MAE_RATIO_MAX:
        return 0.0
    return float(min(1.0, r2 / 0.60))


def _extrapolation_weight(price: float, p_lo: float, p_hi: float) -> float:
    """Снижает вес модели для цен за пределами исторического диапазона."""
    if p_lo >= p_hi:
        return 0.0
    if p_lo <= price <= p_hi:
        return 1.0
    range_width = p_hi - p_lo
    dist = (p_lo - price) / range_width if price < p_lo else (price - p_hi) / range_width
    return float(max(0.0, 1.0 - dist / LGBM_EXTRAP_DECAY_WIDTH))


def fit_lightgbm_sales_model(df: pd.DataFrame, tune: bool = False) -> LGBMModelPack:
    if LGBMRegressor is None:
        return LGBMModelPack(None, [], False, ["Пакет lightgbm не установлен."], 1.0, 1.0, trust_weight=0.0)
    prep = _build_lgbm_training_frame(df)
    if prep.empty:
        return LGBMModelPack(None, [], False, ["Недостаточно валидных строк после очистки."], 1.0, 1.0, trust_weight=0.0)
    warnings = _lgbm_data_warnings(prep)
    p_min = float(prep["our_price"].quantile(LGBM_PRICE_QUANTILE_LO)) if len(prep) else 1.0
    p_max = float(prep["our_price"].quantile(LGBM_PRICE_QUANTILE_HI)) if len(prep) else max(1.0, p_min)
    if p_max <= p_min:
        p_max = max(p_min + 1.0, p_max)

    if warnings:
        # Для слабых данных лучше сразу откатиться, чем обучать нестабильную модель.
        return LGBMModelPack(None, [], False, warnings, p_min, p_max, trust_weight=0.0)
    features = [
        "our_price", "competitor_price", "price_gap", "doy_sin", "doy_cos",
        "dow", "month", "sales_lag1", "sales_lag7", "sales_roll7",
    ]

    split_idx = int(len(prep) * (1.0 - LGBM_HOLDOUT_RATIO))
    split_idx = max(1, min(split_idx, len(prep)))
    df_train = prep.iloc[:split_idx]
    df_hold = prep.iloc[split_idx:]
    can_validate = len(df_hold) >= LGBM_HOLDOUT_MIN_ROWS

    X_train = df_train[features]
    y_train = df_train["sales"].astype(float)
    params = _tune_lgbm_params(X_train, y_train) if tune else LGBM_DEFAULT_PARAMS
    model = LGBMRegressor(random_state=SEED, verbose=-1, **params)
    model.fit(X_train, y_train)

    r2_holdout: float | None = None
    mae_ratio: float | None = None
    trust_weight = 1.0
    reliable = True

    if can_validate:
        from sklearn.metrics import mean_absolute_error

        preds = model.predict(df_hold[features])
        y_hold = df_hold["sales"].astype(float)
        r2_holdout = float(r2_score(y_hold, preds))
        mae = float(mean_absolute_error(y_hold, preds))
        mean_sales = float(y_hold.mean())
        mae_ratio = mae / mean_sales if mean_sales > 0 else 999.0
        trust_weight = _lgbm_trust_weight(r2_holdout, mae_ratio)
        reliable = trust_weight > 0.0
    else:
        warnings.append("holdout_too_small:cannot_validate_quality")

    return LGBMModelPack(
        model=model,
        features=features,
        reliable=reliable,
        warnings=warnings,
        p_min=p_min,
        p_max=p_max,
        trust_weight=trust_weight,
        r2_holdout=r2_holdout,
        mae_ratio=mae_ratio,
    )


def recommend_price_lightgbm(
    history_df: pd.DataFrame,
    next_date: pd.Timestamp,
    last_price: float,
    competitor_price: float,
    cogs: float,
    max_daily_price_change_pct: float = 2.0,
) -> dict:
    model_pack = fit_lightgbm_sales_model(history_df)
    return recommend_price_lightgbm_with_pack(
        model_pack=model_pack,
        history_df=history_df,
        next_date=next_date,
        last_price=last_price,
        competitor_price=competitor_price,
        cogs=cogs,
        max_daily_price_change_pct=max_daily_price_change_pct,
    )


def recommend_price_lightgbm_with_pack(
    model_pack: LGBMModelPack,
    history_df: pd.DataFrame,
    next_date: pd.Timestamp,
    last_price: float,
    competitor_price: float,
    cogs: float,
    max_daily_price_change_pct: float = 2.0,
) -> dict:
    step = abs(float(max_daily_price_change_pct)) / 100.0
    lower = max(1.0, last_price * (1.0 - step))
    upper = max(lower, last_price * (1.0 + step))
    if model_pack.model is None:
        return {
            "recommended_price": float(last_price),
            "pred_sales": float(history_df["sales"].tail(7).mean()) if len(history_df) else 0.0,
            "reliable": False,
            "warnings": model_pack.warnings,
        }
    p_low = max(lower, model_pack.p_min * 0.9, cogs * 1.03)
    p_high = min(upper, model_pack.p_max * 1.1 if model_pack.p_max > 0 else upper)
    if p_high <= p_low:
        p_high = max(p_low * 1.01, upper)
    candidates = np.linspace(p_low, p_high, 21)
    last_sales = float(history_df["sales"].iloc[-1]) if len(history_df) else 0.0
    lag7 = float(history_df["sales"].tail(7).mean()) if len(history_df) else 0.0
    day_of_year = float(next_date.dayofyear)
    base = pd.DataFrame({
        "our_price": candidates,
        "competitor_price": np.full_like(candidates, competitor_price),
        "price_gap": candidates - competitor_price,
        "doy_sin": np.sin(2 * np.pi * day_of_year / 365.0),
        "doy_cos": np.cos(2 * np.pi * day_of_year / 365.0),
        "dow": int(next_date.dayofweek),
        "month": int(next_date.month),
        "sales_lag1": np.full_like(candidates, last_sales),
        "sales_lag7": np.full_like(candidates, lag7),
        "sales_roll7": np.full_like(candidates, lag7),
    })
    pred_sales = np.maximum(0.0, model_pack.model.predict(base[model_pack.features]))
    profits = (candidates - cogs) * pred_sales
    best_i = int(np.argmax(profits))
    return {
        "recommended_price": float(candidates[best_i]),
        "pred_sales": float(pred_sales[best_i]),
        "reliable": model_pack.reliable,
        "warnings": model_pack.warnings,
    }


def predict_sales_lightgbm_with_pack(
    model_pack: LGBMModelPack,
    history_df: pd.DataFrame,
    next_date: pd.Timestamp,
    price: float,
    competitor_price: float,
) -> float:
    fallback = float(history_df["sales"].tail(7).mean()) if len(history_df) else 0.0
    if model_pack.model is None:
        return fallback
    w_quality = float(model_pack.trust_weight)
    w_range = _extrapolation_weight(float(price), float(model_pack.p_min), float(model_pack.p_max))
    w_final = w_quality * w_range

    last_sales = float(history_df["sales"].iloc[-1]) if len(history_df) else 0.0
    lag7 = float(history_df["sales"].tail(7).mean()) if len(history_df) else 0.0
    day_of_year = float(next_date.dayofyear)
    row = pd.DataFrame({
        "our_price": [float(price)],
        "competitor_price": [float(competitor_price)],
        "price_gap": [float(price) - float(competitor_price)],
        "doy_sin": [np.sin(2 * np.pi * day_of_year / 365.0)],
        "doy_cos": [np.cos(2 * np.pi * day_of_year / 365.0)],
        "dow": [int(next_date.dayofweek)],
        "month": [int(next_date.month)],
        "sales_lag1": [last_sales],
        "sales_lag7": [lag7],
        "sales_roll7": [lag7],
    })
    lgbm_pred = float(max(0.0, model_pack.model.predict(row[model_pack.features])[0]))
    demand = w_final * lgbm_pred + (1.0 - w_final) * fallback
    return float(max(0.0, demand))

# ############################################################################
# 2. ЭВРИСТИЧЕСКОЕ ЦЕНООБРАЗОВАНИЕ (RULE-BASED)
# ############################################################################

def apply_rules(row: pd.Series, avg_sales_7d: float) -> tuple[float, str]:
    """
    Применяет бизнес-правила (через динамический RuleEngine) к текущему состоянию рынка для рекомендации цены.
    """
    price = float(row['our_price'])
    comp_1 = float(row.get('competitor_1_price', row.get('competitor_price', price)))
    comp_2 = float(row.get('competitor_2_price', row.get('competitor_price', price)))
    comp_price = min(comp_1, comp_2)
    sales = float(row['sales'])
    cogs = float(row.get('cogs', 0.0))

    # Собираем контекст для движка правил
    context = {
        "price": price,
        "comp_1": comp_1,
        "comp_2": comp_2,
        "comp_price": comp_price,
        "sales": sales,
        "avg_sales_7d": avg_sales_7d,
        "cogs": cogs
    }

    # Выполняем оценку через закешированный экземпляр движка
    engine = get_rule_engine()
    # Если правила обновились в другом процессе, перезагружаем при каждом вызове не очень производительно,
    # Но для MVP достаточно брать то, что есть в engine.rules (мы будем обновлять его внутри приложения)
    return engine.evaluate(context)


# ############################################################################
# 3. МАТЕМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ И РЕГРЕССИЯ
# ############################################################################


def _ols_fit_diagnostics(our_prices: np.ndarray, sales: np.ndarray, min_points: int = MIN_POINTS_FOR_RELIABLE_FIT) -> dict:
    """
    Чистая диагностика OLS Sales~Price, без бизнес-логики откатов/клиппинга.
    Используется и в _fit_linear_sales_vs_price (бинарный reliable),
    и в simulate() для непрерывного веса доверия (blending).
    """
    if len(our_prices) < min_points:
        return {"a": 0.0, "b": 0.0, "coef": 0.0, "r2": 0.0, "n": int(len(our_prices))}
    X = our_prices.reshape(-1, 1)
    y = sales
    model = LinearRegression().fit(X, y)
    a = float(model.intercept_)
    coef = float(model.coef_[0])
    r2 = float(r2_score(y, model.predict(X)))
    b = -coef if coef < 0 else 0.0
    return {"a": a, "b": b, "coef": coef, "r2": r2, "n": int(len(our_prices))}


def _confidence_weight(r2: float, coef: float, min_r2: float = MIN_R2_FOR_RELIABLE_FIT) -> float:
    """
    Непрерывный вес доверия регрессии в [0, 1] вместо бинарного reliable.
    0 = эластичность не отрицательная (или данных мало) → полностью на эвристику.
    1 = R^2 >= min_r2 → полностью на регрессию.
    Между порогами — плавное смешивание (blending), чтобы цена/спрос не
    "дёргались" на границе порога надёжности.
    """
    if coef >= 0 or min_r2 <= 0:
        return 0.0
    return float(np.clip(r2 / min_r2, 0.0, 1.0))


def _fit_linear_sales_vs_price(
    our_prices: np.ndarray,
    sales: np.ndarray,
    fallback_price: float,
    cogs: float = 0.0,
    min_points: int = MIN_POINTS_FOR_RELIABLE_FIT,
    min_r2: float = MIN_R2_FOR_RELIABLE_FIT,
) -> tuple[float, float, float, bool]:
    """
    Общая регрессия: Sales ≈ A - B*Price (B > 0 при надежной отрицательной эластичности).
    Оптимум прибыли Profit = (P - C)*(A - B*P): P* = (A + B*C)/(2B).

    "Надежность" требует не только отрицательного коэффициента, но и
    минимального качества фита (R^2) на тех же точках, плюс минимального числа
    наблюдений. Проверка только по знаку коэффициента пропускала модели,
    которые на практике предсказывают хуже, чем константа (R^2 < 0) —
    это подтвердилось на внешнем датасете (см. README/TESTS, раздел про
    валидацию на Retail Price Optimization).

    Бинарный reliable здесь сохранён для обратной совместимости вызовов
    (fit_regression и т.п.). Для непрерывного смешивания (blending) в
    симуляции используйте _ols_fit_diagnostics + _confidence_weight.
    """
    diag = _ols_fit_diagnostics(our_prices, sales, min_points=min_points)
    a, b, coef, fit_r2 = diag["a"], diag["b"], diag["coef"], diag["r2"]
    is_reliable = (coef < 0) and (fit_r2 >= min_r2)
    if b > 0:
        raw_opt = float((a + b * cogs) / (2 * b))
        p_min_obs = float(np.min(our_prices))
        p_max_obs = float(np.max(our_prices))
        # Ограничиваем оптимум реалистичным коридором вокруг наблюдаемого диапазона.
        p_floor = max(1.0, cogs * 1.03, p_min_obs * 0.85)
        p_ceiling = max(p_floor, p_max_obs * 1.25)
        optimal_price = round(float(np.clip(raw_opt, p_floor, p_ceiling)), 2)
    else:
        optimal_price = float(fallback_price)
    return a, b, optimal_price, is_reliable


def fit_regression(df: pd.DataFrame, product: str) -> tuple[float, float, float, bool]:
    """
    Обучает линейную регрессию Sales = A - B * Price для одного товара (SKU).
    Вычисляет оптимальную цену через экстремум функции прибыли (с учетом COGS).
    """
    prod_data = df[df['product'] == product].copy()
    if "is_oos" in prod_data.columns:
        prod_data = prod_data[~parse_is_oos_series(prod_data["is_oos"])]
    prod_data = prod_data[prod_data["sales"] > 0]
    if len(prod_data) < 3:
        fallback = float(prod_data['our_price'].mean()) if len(prod_data) else 0.0
        return 0.0, 0.0, fallback, False

    c_val = float(prod_data['cogs'].mean()) if 'cogs' in prod_data.columns else PRODUCTS.get(product, {}).get('cogs', 0.0)

    return _fit_linear_sales_vs_price(
        prod_data['our_price'].values.astype(float),
        prod_data['sales'].values.astype(float),
        float(prod_data['our_price'].mean()),
        c_val
    )


def fit_regression_aggregate_daily(work_df: pd.DataFrame) -> tuple[float, float, float, bool]:
    """
    Регрессия по дневному портфелю: одна точка на день.
    Передаем средний cogs для правильного расчета агрегированного оптимума.
    """
    clean_df = work_df.copy()
    if "is_oos" in clean_df.columns:
        clean_df = clean_df[~parse_is_oos_series(clean_df["is_oos"])]
    clean_df = clean_df[clean_df["sales"] > 0]
    if len(clean_df) < 3:
        fb = float(work_df['our_price'].mean()) if len(work_df) else 0.0
        return 0.0, 0.0, fb, False

    c_val = float(clean_df['cogs'].mean()) if 'cogs' in clean_df.columns else 0.0

    return _fit_linear_sales_vs_price(
        clean_df['our_price'].values.astype(float),
        clean_df['sales'].values.astype(float),
        float(clean_df['our_price'].mean()),
        c_val
    )


def predict_sales_regression(a: float, b: float, price: float, noise: float = 0.0) -> int:
    """Спрос по линии регрессии (как в прогнозе рекомендаций), с опциональным шумом."""
    return max(0, int(round(a - b * float(price) + float(noise))))


def _stable_product_id(name: str) -> int:
    return int(zlib.crc32(name.encode("utf-8")) % 900_000) + 100_000


def infer_entity_params(hist: pd.DataFrame, product: str | None = None) -> dict:
    """
    Оценка base_price, base_sales, elasticity и cogs по истории сущности (для загруженного CSV).
    PRODUCTS используется только как запасной вариант при пустой истории.
    """
    if product is None and "product" in hist.columns and not hist.empty:
        product = str(hist["product"].iloc[0])

    work = hist.copy()
    if "is_oos" in work.columns:
        work = work[~parse_is_oos_series(work["is_oos"])]
    work = work[work["sales"] > 0]

    if work.empty:
        if product and product in PRODUCTS:
            return dict(PRODUCTS[product])
        return {
            "id": _stable_product_id(product or ""),
            "base_price": 0.0,
            "base_sales": 0.0,
            "elasticity": DEFAULT_ELASTICITY,
            "cogs": 0.0,
        }

    base_price = float(work["our_price"].median())
    base_sales = float(work["sales"].median())
    cogs = float(work["cogs"].mean()) if "cogs" in work.columns else 0.0

    if "product_id" in work.columns and work["product_id"].notna().any():
        pid = int(work["product_id"].iloc[-1])
    elif product and product in PRODUCTS:
        pid = int(PRODUCTS[product]["id"])
    else:
        pid = _stable_product_id(str(product or ""))

    elasticity = DEFAULT_ELASTICITY
    if len(work) >= 3:
        _, b, _, reliable = _fit_linear_sales_vs_price(
            work["our_price"].values.astype(float),
            work["sales"].values.astype(float),
            base_price,
            cogs,
        )
        if reliable and b > 0:
            elasticity = max(0.2, float(b))

    return {
        "id": pid,
        "base_price": base_price,
        "base_sales": base_sales,
        "elasticity": elasticity,
        "cogs": cogs,
    }


def products_in_dataframe(df: pd.DataFrame) -> list[str]:
    """Уникальные SKU из df (стабильная сортировка по имени)."""
    if df is None or df.empty or "product" not in df.columns:
        return []
    names = df["product"].dropna().astype(str).str.strip()
    return sorted(n for n in names.unique().tolist() if n)


def forecast(
    product: str,
    recommended_price: float,
    current_metric: float,
    regression_params: tuple = None,
    product_params: dict | None = None,
) -> dict:
    """
    Прогноз выручки и прибыли (DRY: заменяет две старые функции).
    Если переданы regression_params = (a, b), используется регрессия.
    Иначе — эвристика по product_params или PRODUCTS.
    """
    p = product_params if product_params is not None else PRODUCTS.get(product, {})
    cogs = p.get("cogs", 0)

    if regression_params:
        a, b = regression_params
        pred_sales = max(0, round(a - b * recommended_price))
    else:
        price_dev = recommended_price - p.get("base_price", 0)
        pred_sales = max(0, round(p.get("base_sales", 0) - p.get("elasticity", 0) * price_dev))

    pred_revenue = pred_sales * recommended_price
    pred_profit = pred_sales * (recommended_price - cogs)

    growth_pct = ((pred_profit - current_metric) / current_metric * 100) if current_metric > 0 else (0.0 if pred_profit == 0 else 100.0)
    return {
        'forecast_sales': pred_sales,
        'forecast_revenue': round(pred_revenue, 2),
        'forecast_profit': round(pred_profit, 2),
        'growth_pct': round(growth_pct, 1)
    }


# ############################################################################
# 4. МОДУЛЬ СИМУЛЯЦИИ (TIME-ROLL FORWARD)
# ############################################################################


def simulate(
    df: pd.DataFrame,
    n_steps: int,
    method: str,
    target_product: str = None,
    retrain_every_days: int = 7,
    train_window_days: int = 90,
    max_daily_price_change_pct: float = 2.0,
) -> pd.DataFrame:
    """
    Запускает циклическую симуляцию рынка.
    methods: rules | regression | lightgbm
    """
    sim_df = df.copy()
    history_days = int(sim_df["date"].nunique()) if "date" in sim_df.columns and len(sim_df) > 0 else 0
    if history_days > 0:
        train_window_days = int(np.clip(int(train_window_days), 1, history_days))
    else:
        train_window_days = 1

    if target_product and target_product != "Все товары":
        present_products = [target_product]
    else:
        present_products = products_in_dataframe(sim_df)

    if not present_products:
        return sim_df

    hierarchy_cols = [col for col in ['store_id', 'store', 'store_profile', 'brand_id', 'brand'] if col in sim_df.columns]
    entity_cols = hierarchy_cols + ['product_id', 'product'] if 'product_id' in sim_df.columns else hierarchy_cols + ['product']
    entities_df = (
        sim_df[sim_df['product'].isin(present_products)][entity_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    n_entities = len(entities_df)
    if n_entities == 0:
        return sim_df

    products_by_entity = entities_df["product"].tolist()
    base_prices = np.zeros(n_entities, dtype=float)
    base_sales = np.zeros(n_entities, dtype=float)
    elasticities = np.zeros(n_entities, dtype=float)
    cogs = np.zeros(n_entities, dtype=float)
    product_ids = np.zeros(n_entities, dtype=int)

    reg_a = np.zeros(n_entities, dtype=float)
    reg_b = np.zeros(n_entities, dtype=float)
    reg_rel = np.zeros(n_entities, dtype=bool)
    # Непрерывный вес доверия (0..1) вместо бинарного reg_rel — используется для
    # плавного смешивания (blending) цены и спроса между регрессией и эвристикой,
    # чтобы не было резкого скачка на границе порога надёжности.
    reg_weight = np.zeros(n_entities, dtype=float)
    reg_target_prices = np.zeros(n_entities, dtype=float)
    season_phase = np.array([SEASONALITY_PHASE.get(p, 0.0) for p in products_by_entity], dtype=float)

    last_our_prices = np.zeros(n_entities)
    last_comp_1_prices = np.zeros(n_entities)
    last_comp_2_prices = np.zeros(n_entities)
    last_sales = np.zeros(n_entities)
    sales_buffer = np.zeros((7, n_entities))
    history_prices: list[np.ndarray] = [np.array([], dtype=float) for _ in range(n_entities)]
    history_sales: list[np.ndarray] = [np.array([], dtype=float) for _ in range(n_entities)]
    history_oos: list[np.ndarray] = [np.array([], dtype=bool) for _ in range(n_entities)]
    history_comp_prices: list[np.ndarray] = [np.array([], dtype=float) for _ in range(n_entities)]
    history_dates: list[np.ndarray] = [np.array([], dtype="datetime64[ns]") for _ in range(n_entities)]
    lgbm_packs_by_entity: list[LGBMModelPack] = [
        LGBMModelPack(None, [], False, ["Модель еще не обучалась."], 1.0, 1.0)
        for _ in range(n_entities)
    ]

    for i, entity in entities_df.iterrows():
        mask = sim_df['product'] == entity['product']
        for h_col in hierarchy_cols:
            mask &= sim_df[h_col] == entity[h_col]
        hist = sim_df[mask]
        params = infer_entity_params(hist, str(entity["product"]))
        base_prices[i] = float(params["base_price"])
        base_sales[i] = float(params["base_sales"])
        elasticities[i] = float(params["elasticity"])
        cogs[i] = float(params.get("cogs", 0.0))
        product_ids[i] = int(params["id"])
        if not hist.empty:
            last_our = float(hist["our_price"].iloc[-1])
            last_our_prices[i] = last_our
            if "competitor_price" in hist.columns:
                comp_ref = hist["competitor_price"]
            else:
                comp_ref = hist["our_price"]
            last_comp_1_prices[i] = float(
                hist["competitor_1_price"].iloc[-1] if "competitor_1_price" in hist.columns else comp_ref.iloc[-1]
            )
            last_comp_2_prices[i] = float(
                hist["competitor_2_price"].iloc[-1] if "competitor_2_price" in hist.columns else comp_ref.iloc[-1]
            )
            last_sales[i] = float(hist['sales'].iloc[-1])
            hs_vals = hist['sales'].values
            if len(hs_vals) >= 7:
                sales_buffer[:, i] = hs_vals[-7:]
            else:
                sales_buffer[-len(hs_vals):, i] = hs_vals
                sales_buffer[:-len(hs_vals), i] = hs_vals.mean() if len(hs_vals) > 0 else 0
            history_prices[i] = hist['our_price'].to_numpy(dtype=float)
            history_sales[i] = hist['sales'].to_numpy(dtype=float)
            if "competitor_price" in hist.columns:
                history_comp_prices[i] = hist["competitor_price"].to_numpy(dtype=float)
            else:
                history_comp_prices[i] = hist["our_price"].to_numpy(dtype=float)
            history_dates[i] = pd.to_datetime(hist['date']).to_numpy(dtype="datetime64[ns]")
            if 'is_oos' in hist.columns:
                history_oos[i] = parse_is_oos_series(hist["is_oos"]).to_numpy()
            else:
                history_oos[i] = np.zeros(len(hist), dtype=bool)
        reg_target_prices[i] = last_our_prices[i]

    rng = np.random.default_rng(SEED)
    last_date = sim_df['date'].max()
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)

    all_dates = np.repeat(date_range.values, n_entities)
    all_ids = np.tile(product_ids, n_steps)
    all_names = np.tile(np.array(products_by_entity), n_steps)
    hierarchy_values = {col: np.tile(entities_df[col].to_numpy(), n_steps) for col in hierarchy_cols}

    out_our_prices = np.zeros((n_steps, n_entities))
    out_comp_1_prices = np.zeros((n_steps, n_entities))
    out_comp_2_prices = np.zeros((n_steps, n_entities))
    out_comp_prices = np.zeros((n_steps, n_entities))
    out_sales = np.zeros((n_steps, n_entities))
    out_revenue = np.zeros((n_steps, n_entities))
    out_profit = np.zeros((n_steps, n_entities))
    out_cogs = np.tile(cogs, (n_steps, 1))
    out_oos = np.zeros((n_steps, n_entities), dtype=bool)
    lgbm_pred_sales = np.zeros(n_entities, dtype=float)
    aggressive_mask = rng.random(n_entities) < 0.45

    for step in range(n_steps):
        rec_prices = np.zeros(n_entities)
        day_of_year = pd.Timestamp(date_range[step]).dayofyear
        seasonal_multiplier = 1.0 + 0.16 * np.sin((2 * np.pi / 365.0) * day_of_year + season_phase)
        seasonal_multiplier = np.maximum(0.2, seasonal_multiplier)
        seasonal_base_sales = np.maximum(0.0, base_sales * seasonal_multiplier)

        if method == 'rules':
            # Для правил нужно опираться на среднее за последние 7 дней
            avg7 = sales_buffer.mean(axis=0)
            for i, p in enumerate(products_by_entity):
                row_mock = pd.Series({
                    'our_price': last_our_prices[i],
                    'competitor_1_price': last_comp_1_prices[i],
                    'competitor_2_price': last_comp_2_prices[i],
                    'competitor_price': min(last_comp_1_prices[i], last_comp_2_prices[i]),
                    'sales': last_sales[i]
                })
                rp, _ = apply_rules(row_mock, avg7[i])
                rec_prices[i] = rp
        else:
            retrain_step = (step == 0) or (retrain_every_days > 0 and step % retrain_every_days == 0)
            if retrain_step:
                for i in range(n_entities):
                    if method == 'regression':
                        hp = history_prices[i][-train_window_days:] if train_window_days > 0 else history_prices[i]
                        hs = history_sales[i][-train_window_days:] if train_window_days > 0 else history_sales[i]
                        ho = history_oos[i][-len(hp):] if len(hp) > 0 else np.array([], dtype=bool)
                        valid_mask = (~ho) & (hs > 0)
                        if np.sum(valid_mask) >= 3:
                            diag_i = _ols_fit_diagnostics(hp[valid_mask], hs[valid_mask])
                            a_i, b_i, coef_i, r2_i = diag_i["a"], diag_i["b"], diag_i["coef"], diag_i["r2"]
                            rel_i = (coef_i < 0) and (r2_i >= MIN_R2_FOR_RELIABLE_FIT)
                            w_i = _confidence_weight(r2_i, coef_i)
                            if b_i > 0:
                                raw_opt = float((a_i + b_i * float(cogs[i])) / (2 * b_i))
                                p_min_obs = float(np.min(hp[valid_mask]))
                                p_max_obs = float(np.max(hp[valid_mask]))
                                p_floor = max(1.0, float(cogs[i]) * 1.03, p_min_obs * 0.85)
                                p_ceiling = max(p_floor, p_max_obs * 1.25)
                                opt_i = round(float(np.clip(raw_opt, p_floor, p_ceiling)), 2)
                            else:
                                opt_i = float(last_our_prices[i])
                        else:
                            a_i, b_i, opt_i, rel_i, w_i = 0.0, 0.0, float(last_our_prices[i]), False, 0.0
                        reg_a[i] = a_i
                        reg_b[i] = b_i
                        reg_rel[i] = rel_i
                        reg_weight[i] = w_i
                        # Смешиваем оптимум регрессии с текущей ценой по весу доверия,
                        # а не переключаемся бинарно — убирает скачок на границе порога.
                        target_price = w_i * float(opt_i) + (1.0 - w_i) * float(last_our_prices[i])
                        step_limit = abs(float(max_daily_price_change_pct)) / 100.0
                        lower = max(1.0, float(last_our_prices[i]) * (1.0 - step_limit))
                        upper = max(lower, float(last_our_prices[i]) * (1.0 + step_limit))
                        reg_target_prices[i] = float(np.clip(target_price, lower, upper))
                    elif method == 'lightgbm':
                        pass
            if method == 'lightgbm' and retrain_step:
                lgbm_packs_by_entity = []
                for i in range(n_entities):
                    hw = slice(-train_window_days, None) if train_window_days > 0 else slice(None)
                    n_i = len(history_prices[i][hw])
                    if n_i == 0:
                        lgbm_packs_by_entity.append(
                            LGBMModelPack(None, [], False, ["Нет истории для обучения."], 1.0, 1.0)
                        )
                        continue
                    hist_df = pd.DataFrame({
                        "date": history_dates[i][hw],
                        "our_price": history_prices[i][hw],
                        "competitor_price": history_comp_prices[i][hw],
                        "sales": history_sales[i][hw],
                        "is_oos": history_oos[i][hw],
                        "cogs": np.full(n_i, float(cogs[i])),
                    }).sort_values("date")
                    lgbm_packs_by_entity.append(fit_lightgbm_sales_model(hist_df, tune=True))
            if method == 'lightgbm':
                for i in range(n_entities):
                    # Экономим время симуляции: для инференса из истории нужны только последние продажи для lags
                    tail_len = min(7, len(history_sales[i]))
                    hist_df = pd.DataFrame({
                        "sales": history_sales[i][-tail_len:]
                    })
                    if i >= len(lgbm_packs_by_entity):
                        logger.warning(f"Индекс {i} выходит за границы lgbm_packs_by_entity (размер: {len(lgbm_packs_by_entity)})")
                        pack = LGBMModelPack(None, [], False, ["Модель не найдена."], 1.0, 1.0)
                    else:
                        pack = lgbm_packs_by_entity[i]
                    lgbm_rec = recommend_price_lightgbm_with_pack(
                        model_pack=pack,
                        history_df=hist_df,
                        next_date=pd.Timestamp(date_range[step]),
                        last_price=float(last_our_prices[i]),
                        competitor_price=float(min(last_comp_1_prices[i], last_comp_2_prices[i])),
                        cogs=float(cogs[i]),
                        max_daily_price_change_pct=float(max_daily_price_change_pct),
                    )
                    reg_target_prices[i] = float(lgbm_rec["recommended_price"])
                    lgbm_pred_sales[i] = float(lgbm_rec["pred_sales"])
                    reg_rel[i] = bool(lgbm_rec["reliable"])
            rec_prices[:] = reg_target_prices

        comp1_base = base_prices * 1.03
        comp2_base = base_prices * 0.93
        noise_comp_1 = rng.normal(0, 0.015 * base_prices, size=n_entities)
        noise_comp_2 = rng.normal(0, 0.020 * base_prices, size=n_entities)
        chaos_step = aggressive_mask & (rng.random(n_entities) < 0.15)
        competitor_1_prices = calc_competitor_1_prices(last_comp_1_prices, comp1_base, last_our_prices, noise_comp_1)
        competitor_2_prices = calc_competitor_2_prices(
            last_comp_2_prices, comp2_base, base_prices * 0.90, noise_comp_2, chaos_step
        )
        competitor_prices = np.minimum(competitor_1_prices, competitor_2_prices)

        new_sales = np.zeros(n_entities)

        if method == 'regression':
            # Раньше было бинарное разделение (valid_mask = reg_rel & (reg_b > 0)):
            # сущность целиком на регрессии ИЛИ целиком на эвристике. Теперь считаем
            # оба прогноза для всех и смешиваем по непрерывному весу доверия reg_weight —
            # это устраняет резкий скачок прогноза на границе порога надёжности.
            noise_scale_reg = np.maximum(2.0, 0.08 * np.maximum(np.abs(reg_a - reg_b * rec_prices), 1.0))
            sales_reg = calc_demand_regression(
                rec_prices, reg_a, reg_b, rng.normal(0, noise_scale_reg, size=n_entities)
            )

            noise_rules_all = rng.normal(0, np.maximum(2.0, 0.08 * seasonal_base_sales))
            sales_rules = calc_demand_rules(
                rec_prices, competitor_prices, base_prices, seasonal_base_sales, elasticities, noise_rules_all
            )

            w = np.where(reg_b > 0, reg_weight, 0.0)
            new_sales = w * sales_reg + (1.0 - w) * sales_rules
            # Для регрессионной ветки также учитываем сезонный множитель спроса.
            new_sales = np.maximum(0.0, np.round(new_sales * seasonal_multiplier))
        elif method == 'lightgbm':
            for i in range(n_entities):
                if reg_rel[i]:
                    new_sales[i] = max(0.0, round(lgbm_pred_sales[i]))
                else:
                    noise_rules = rng.normal(0, max(2.0, 0.08 * seasonal_base_sales[i]))
                    new_sales[i] = calc_demand_rules(
                        np.array([rec_prices[i]]),
                        np.array([competitor_prices[i]]),
                        np.array([base_prices[i]]),
                        np.array([seasonal_base_sales[i]]),
                        np.array([elasticities[i]]),
                        np.array([noise_rules]),
                    )[0]
            new_sales = np.maximum(0.0, np.round(new_sales))
        else:
            # Чистая эвристика
            noise = rng.normal(0, np.maximum(2.0, 0.08 * seasonal_base_sales), size=n_entities)
            new_sales = calc_demand_rules(
                rec_prices, competitor_prices, base_prices, seasonal_base_sales, elasticities, noise
            )
        oos_mask = rng.random(n_entities) < 0.05
        new_sales[oos_mask] = 0.0

        revenue = np.round(new_sales * rec_prices, 2)
        profit = np.round(revenue - new_sales * cogs, 2)

        # Обновляем state
        last_our_prices = rec_prices.copy()
        last_comp_1_prices = competitor_1_prices.copy()
        last_comp_2_prices = competitor_2_prices.copy()
        last_sales = new_sales.copy()

        sales_buffer[:-1, :] = sales_buffer[1:, :]
        sales_buffer[-1, :] = new_sales

        # Сохраняем в матрицы шага
        out_our_prices[step, :] = rec_prices
        out_comp_1_prices[step, :] = competitor_1_prices
        out_comp_2_prices[step, :] = competitor_2_prices
        out_comp_prices[step, :] = competitor_prices
        out_sales[step, :] = new_sales
        out_revenue[step, :] = revenue
        out_profit[step, :] = profit
        out_oos[step, :] = oos_mask
        for i in range(n_entities):
            history_prices[i] = np.append(history_prices[i], rec_prices[i])
            history_comp_prices[i] = np.append(history_comp_prices[i], competitor_prices[i])
            history_sales[i] = np.append(history_sales[i], new_sales[i])
            history_oos[i] = np.append(history_oos[i], bool(oos_mask[i]))
            history_dates[i] = np.append(history_dates[i], np.array([date_range[step]], dtype="datetime64[ns]"))
            keep_len = max(14, train_window_days * 2)
            if len(history_prices[i]) > keep_len:
                history_prices[i] = history_prices[i][-keep_len:]
                history_comp_prices[i] = history_comp_prices[i][-keep_len:]
                history_sales[i] = history_sales[i][-keep_len:]
                history_oos[i] = history_oos[i][-keep_len:]
                history_dates[i] = history_dates[i][-keep_len:]

    # Собираем результат воедино
    new_data = {
        'date': all_dates,
        'product_id': all_ids,
        'product': all_names,
        'our_price': out_our_prices.flatten(),
        'competitor_1_price': out_comp_1_prices.flatten(),
        'competitor_2_price': out_comp_2_prices.flatten(),
        'competitor_price': out_comp_prices.flatten(),
        'is_oos': out_oos.flatten(),
        'sales': out_sales.flatten(),
        'revenue': out_revenue.flatten(),
        'cogs': out_cogs.flatten(),
        'profit': out_profit.flatten(),
    }
    for col in hierarchy_cols:
        new_data[col] = hierarchy_values[col]
    new_df = pd.DataFrame(new_data)

    sim_df = pd.concat([sim_df, new_df], ignore_index=True)
    return sim_df
