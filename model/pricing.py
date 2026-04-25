import numpy as np
import pandas as pd
from dataclasses import dataclass

from model.math_engine import (
    calc_competitor_1_prices,
    calc_competitor_2_prices,
    calc_demand_regression,
    calc_demand_rules,
)
from sklearn.linear_model import LinearRegression
try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - fallback если пакет не установлен
    LGBMRegressor = None

# ############################################################################
# 1. КОНСТАНТЫ И КОНФИГУРАЦИЯ (Единый источник правды)
# ############################################################################

SEED = 42

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
LGBM_MIN_PRICE_STD = 1.0


@dataclass
class LGBMModelPack:
    model: object | None
    features: list[str]
    reliable: bool
    warnings: list[str]
    p_min: float
    p_max: float


def _lgbm_data_warnings(df: pd.DataFrame) -> list[str]:
    msgs: list[str] = []
    n_rows = len(df)
    if n_rows < LGBM_MIN_ROWS:
        msgs.append(f"Наблюдений мало: {n_rows} (< {LGBM_MIN_ROWS}).")
    uniq_prices = int(df["our_price"].nunique()) if "our_price" in df.columns and n_rows > 0 else 0
    if uniq_prices < LGBM_MIN_UNIQUE_PRICES:
        msgs.append(f"Слабая вариативность цены: уникальных значений {uniq_prices} (< {LGBM_MIN_UNIQUE_PRICES}).")
    if n_rows > 1:
        price_std = float(df["our_price"].std(ddof=0))
        if price_std < LGBM_MIN_PRICE_STD:
            msgs.append(f"Слишком узкий диапазон цен: std={price_std:.2f}.")
    return msgs


def _build_lgbm_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "is_oos" in work.columns:
        work = work[~work["is_oos"].astype(bool)]
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


def fit_lightgbm_sales_model(df: pd.DataFrame) -> LGBMModelPack:
    if LGBMRegressor is None:
        return LGBMModelPack(None, [], False, ["Пакет lightgbm не установлен."], 1.0, 1.0)
    prep = _build_lgbm_training_frame(df)
    if prep.empty:
        return LGBMModelPack(None, [], False, ["Недостаточно валидных строк после очистки."], 1.0, 1.0)
    warnings = _lgbm_data_warnings(prep)
    if warnings:
        # Для слабых данных лучше сразу откатиться, чем обучать нестабильную модель.
        p_min = float(prep["our_price"].quantile(0.05)) if len(prep) else 1.0
        p_max = float(prep["our_price"].quantile(0.95)) if len(prep) else max(1.0, p_min)
        return LGBMModelPack(None, [], False, warnings, p_min, p_max)
    features = [
        "our_price", "competitor_price", "price_gap", "doy_sin", "doy_cos",
        "dow", "month", "sales_lag1", "sales_lag7", "sales_roll7",
    ]
    X = prep[features]
    y = prep["sales"].astype(float)
    model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X, y)
    return LGBMModelPack(
        model=model,
        features=features,
        reliable=(len(warnings) == 0),
        warnings=warnings,
        p_min=float(prep["our_price"].quantile(0.05)),
        p_max=float(prep["our_price"].quantile(0.95)),
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
    if model_pack.model is None:
        return float(history_df["sales"].tail(7).mean()) if len(history_df) else 0.0
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
    return float(max(0.0, model_pack.model.predict(row[model_pack.features])[0]))

# ############################################################################
# 2. ЭВРИСТИЧЕСКОЕ ЦЕНООБРАЗОВАНИЕ (RULE-BASED)
# ############################################################################

def apply_rules(row: pd.Series, avg_sales_7d: float) -> tuple[float, str]:
    """
    Применяет бизнес-правила к текущему состоянию рынка для рекомендации цены.

    Аргументы:
        row: pd.Series с данными одного дня (our_price, competitor_price, sales).
        avg_sales_7d: Средние продажи за последние 7 дней.

    Возвращает:
        tuple (рекомендованная_цена, название_правила).
    """
    price = float(row['our_price'])
    comp_1 = float(row.get('competitor_1_price', row.get('competitor_price', price)))
    comp_2 = float(row.get('competitor_2_price', row.get('competitor_price', price)))
    comp_price = min(comp_1, comp_2)
    sales = float(row['sales'])

    # ПРАВИЛО 0: Строгое соответствие ТЗ Спринта 2 (если продаж совсем не было)
    if sales == 0:
        return round(price - 1.0, 2), "zero_sales_strict"

    # ПРАВИЛО 1: Реакция на конкурента (Спринт 2)
    # Если конкурент значительно дешевле (на 10% и более), снижаем цену на 5%
    if comp_price < price * 0.90:
        return round(price * 0.95, 2), "competitor_undercut"

    # ПРАВИЛО 2: Реакция на падение спроса (Спринт 2)
    # Если продажи за вчера на 20% ниже скользящего среднего, пробуем стимулировать спрос
    if sales < avg_sales_7d * 0.80:
        return round(price - 1.0, 2), "low_sales"

    # ПРАВИЛО 3: Умеренная реакция на разрыв с конкурентом (более чувствительное)
    if comp_price < price * 0.97:
        return round(max(1, price * 0.98), 2), "competitor_gap_soft"

    # ПРАВИЛО 4: Если мы заметно дешевле конкурента и спрос не проседает, можно слегка поднять цену
    if comp_price > price * 1.07 and sales >= avg_sales_7d * 0.95:
        return round(price * 1.015, 2), "margin_recovery"

    # ПРАВИЛО 5: Сглаживание — маленький шаг к цене конкурента, чтобы не зависать на hold
    midpoint = (price + comp_price) / 2
    if midpoint > price * 1.01:
        return round(price * 1.005, 2), "drift_up"
    if midpoint < price * 0.99:
        return round(max(1, price * 0.995), 2), "drift_down"

    return price, "hold"


# ############################################################################
# 3. МАТЕМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ И РЕГРЕССИЯ
# ############################################################################


def _fit_linear_sales_vs_price(our_prices: np.ndarray, sales: np.ndarray, fallback_price: float, cogs: float = 0.0) -> tuple[float, float, float, bool]:
    """
    Общая регрессия: Sales ≈ A - B*Price (B > 0 при надежной отрицательной эластичности).
    Оптимум прибыли Profit = (P - C)*(A - B*P): P* = (A + B*C)/(2B).
    """
    if len(our_prices) < 3:
        return 0.0, 0.0, float(fallback_price), False

    X = our_prices.reshape(-1, 1)
    y = sales
    model = LinearRegression().fit(X, y)
    a = float(model.intercept_)
    coef = float(model.coef_[0])
    is_reliable = coef < 0
    b = -coef if coef < 0 else 0.0
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
        prod_data = prod_data[~prod_data["is_oos"].astype(bool)]
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
        clean_df = clean_df[~clean_df["is_oos"].astype(bool)]
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


def products_in_dataframe(df: pd.DataFrame) -> list[str]:
    """SKU из PRODUCTS, по которым есть строки в df (устойчивый порядок)."""
    return [p for p in PRODUCTS if not df.loc[df['product'] == p].empty]


def forecast(product: str, recommended_price: float, current_metric: float, regression_params: tuple = None) -> dict:
    """
    Прогноз выручки и прибыли (DRY: заменяет две старые функции).
    Если переданы regression_params = (a, b), используется регрессия. Иначе — базовая эластичность (PRODUCTS).
    """
    p = PRODUCTS.get(product, {})
    cogs = p.get('cogs', 0)

    if regression_params:
        a, b = regression_params
        pred_sales = max(0, round(a - b * recommended_price))
    else:
        price_dev = recommended_price - p.get('base_price', 0)
        pred_sales = max(0, round(p.get('base_sales', 0) - p.get('elasticity', 0) * price_dev))

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
        present_products = [p for p in PRODUCTS if (sim_df['product'] == p).any()]

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

    products_by_entity = entities_df['product'].tolist()
    product_to_indices: dict[str, list[int]] = {}
    for idx, prod in enumerate(products_by_entity):
        product_to_indices.setdefault(prod, []).append(idx)
    base_prices = np.array([PRODUCTS[p]['base_price'] for p in products_by_entity], dtype=float)
    base_sales = np.array([PRODUCTS[p]['base_sales'] for p in products_by_entity], dtype=float)
    elasticities = np.array([PRODUCTS[p]['elasticity'] for p in products_by_entity], dtype=float)
    cogs = np.array([PRODUCTS[p].get('cogs', 0.0) for p in products_by_entity], dtype=float)
    if 'product_id' in entities_df.columns:
        product_ids = entities_df['product_id'].to_numpy()
    else:
        product_ids = np.array([PRODUCTS[p]['id'] for p in products_by_entity])

    reg_a = np.zeros(n_entities, dtype=float)
    reg_b = np.zeros(n_entities, dtype=float)
    reg_rel = np.zeros(n_entities, dtype=bool)
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
    lgbm_packs_by_product: dict[str, LGBMModelPack] = {}

    for i, entity in entities_df.iterrows():
        mask = sim_df['product'] == entity['product']
        for h_col in hierarchy_cols:
            mask &= sim_df[h_col] == entity[h_col]
        hist = sim_df[mask]
        if not hist.empty:
            last_our_prices[i] = float(hist['our_price'].iloc[-1])
            last_comp_1_prices[i] = float(hist.get('competitor_1_price', hist['competitor_price']).iloc[-1])
            last_comp_2_prices[i] = float(hist.get('competitor_2_price', hist['competitor_price']).iloc[-1])
            last_sales[i] = float(hist['sales'].iloc[-1])
            hs_vals = hist['sales'].values
            if len(hs_vals) >= 7:
                sales_buffer[:, i] = hs_vals[-7:]
            else:
                sales_buffer[-len(hs_vals):, i] = hs_vals
                sales_buffer[:-len(hs_vals), i] = hs_vals.mean() if len(hs_vals) > 0 else 0
            history_prices[i] = hist['our_price'].to_numpy(dtype=float)
            history_sales[i] = hist['sales'].to_numpy(dtype=float)
            history_comp_prices[i] = hist.get('competitor_price', hist['our_price']).to_numpy(dtype=float)
            history_dates[i] = pd.to_datetime(hist['date']).to_numpy(dtype="datetime64[ns]")
            if 'is_oos' in hist.columns:
                history_oos[i] = hist['is_oos'].astype(bool).to_numpy()
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
                            a_i, b_i, opt_i, rel_i = _fit_linear_sales_vs_price(
                                hp[valid_mask],
                                hs[valid_mask],
                                float(last_our_prices[i]),
                                float(cogs[i]),
                            )
                        else:
                            a_i, b_i, opt_i, rel_i = 0.0, 0.0, float(last_our_prices[i]), False
                        reg_a[i] = a_i
                        reg_b[i] = b_i
                        reg_rel[i] = rel_i
                        target_price = float(opt_i) if rel_i else float(last_our_prices[i])
                        step_limit = abs(float(max_daily_price_change_pct)) / 100.0
                        lower = max(1.0, float(last_our_prices[i]) * (1.0 - step_limit))
                        upper = max(lower, float(last_our_prices[i]) * (1.0 + step_limit))
                        reg_target_prices[i] = float(np.clip(target_price, lower, upper))
                    elif method == 'lightgbm':
                        pass
            if method == 'lightgbm' and retrain_step:
                lgbm_packs_by_product = {}
                for prod, idxs in product_to_indices.items():
                    date_parts = []
                    price_parts = []
                    comp_parts = []
                    sales_parts = []
                    oos_parts = []
                    cogs_parts = []
                    for i in idxs:
                        hw = slice(-train_window_days, None) if train_window_days > 0 else slice(None)
                        n_i = len(history_prices[i][hw])
                        if n_i == 0:
                            continue
                        date_parts.append(pd.to_datetime(history_dates[i][hw]))
                        price_parts.append(history_prices[i][hw])
                        comp_parts.append(history_comp_prices[i][hw])
                        sales_parts.append(history_sales[i][hw])
                        oos_parts.append(history_oos[i][hw])
                        cogs_parts.append(np.full(n_i, float(cogs[i])))
                    if not price_parts:
                        lgbm_packs_by_product[prod] = LGBMModelPack(None, [], False, ["Нет истории для обучения."], 1.0, 1.0)
                        continue
                    hist_df = pd.DataFrame({
                        "date": pd.concat([pd.Series(x) for x in date_parts], ignore_index=True),
                        "our_price": np.concatenate(price_parts),
                        "competitor_price": np.concatenate(comp_parts),
                        "sales": np.concatenate(sales_parts),
                        "is_oos": np.concatenate(oos_parts),
                        "cogs": np.concatenate(cogs_parts),
                    }).sort_values("date")
                    lgbm_packs_by_product[prod] = fit_lightgbm_sales_model(hist_df)
            if method == 'lightgbm':
                for i in range(n_entities):
                    hw = slice(-train_window_days, None) if train_window_days > 0 else slice(None)
                    hist_df = pd.DataFrame({
                        "date": pd.to_datetime(history_dates[i][hw]),
                        "our_price": history_prices[i][hw],
                        "competitor_price": history_comp_prices[i][hw],
                        "sales": history_sales[i][hw],
                        "is_oos": history_oos[i][hw],
                        "cogs": np.full(len(history_prices[i][hw]), float(cogs[i])),
                    })
                    pack = lgbm_packs_by_product.get(products_by_entity[i], LGBMModelPack(None, [], False, ["Нет модели по товару."], 1.0, 1.0))
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
            valid_mask = reg_rel & (reg_b > 0)

            # Надежная регрессия
            noise_scale_reg = np.maximum(2.0, 0.08 * np.maximum(np.abs(reg_a - reg_b * rec_prices), 1.0))
            new_sales[valid_mask] = calc_demand_regression(
                rec_prices, reg_a, reg_b, rng.normal(0, noise_scale_reg, size=n_entities)
            )[valid_mask]

            # Откат на эвристику (ненадежная регрессия)
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                noise_rules = rng.normal(0, np.maximum(2.0, 0.08 * seasonal_base_sales[invalid_mask]))
                new_sales[invalid_mask] = calc_demand_rules(
                    rec_prices[invalid_mask], competitor_prices[invalid_mask], base_prices[invalid_mask],
                    seasonal_base_sales[invalid_mask], elasticities[invalid_mask], noise_rules
                )
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
            new_sales = np.maximum(0.0, np.round(new_sales * seasonal_multiplier))
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
