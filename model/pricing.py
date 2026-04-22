import numpy as np
import pandas as pd

from model.math_engine import (
    calc_competitor_1_prices,
    calc_competitor_2_prices,
    calc_demand_regression,
    calc_demand_rules,
)
from sklearn.linear_model import LinearRegression

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
    Запускает циклическую симуляцию рынка векторно (NumPy-based).
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
                    # Ограничиваем дневной шаг цены, чтобы избежать нереалистичных скачков.
                    step_limit = abs(float(max_daily_price_change_pct)) / 100.0
                    lower = max(1.0, float(last_our_prices[i]) * (1.0 - step_limit))
                    upper = max(lower, float(last_our_prices[i]) * (1.0 + step_limit))
                    reg_target_prices[i] = float(np.clip(target_price, lower, upper))
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
            history_sales[i] = np.append(history_sales[i], new_sales[i])
            history_oos[i] = np.append(history_oos[i], bool(oos_mask[i]))
            keep_len = max(14, train_window_days * 2)
            if len(history_prices[i]) > keep_len:
                history_prices[i] = history_prices[i][-keep_len:]
                history_sales[i] = history_sales[i][-keep_len:]
                history_oos[i] = history_oos[i][-keep_len:]

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
