import numpy as np
import pandas as pd

from model.math_engine import (
    calc_competitor_prices,
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
    comp_price = float(row['competitor_price'])
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
        optimal_price = round((a + b * cogs) / (2 * b), 2)
    else:
        optimal_price = float(fallback_price)
    return a, b, optimal_price, is_reliable


def fit_regression(df: pd.DataFrame, product: str) -> tuple[float, float, float, bool]:
    """
    Обучает линейную регрессию Sales = A - B * Price для одного товара (SKU).
    Вычисляет оптимальную цену через экстремум функции прибыли (с учетом COGS).
    """
    prod_data = df[df['product'] == product]
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
    if len(work_df) < 3:
        fb = float(work_df['our_price'].mean()) if len(work_df) else 0.0
        return 0.0, 0.0, fb, False
    
    c_val = float(work_df['cogs'].mean()) if 'cogs' in work_df.columns else 0.0
    
    return _fit_linear_sales_vs_price(
        work_df['our_price'].values.astype(float),
        work_df['sales'].values.astype(float),
        float(work_df['our_price'].mean()),
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


def simulate(df: pd.DataFrame, n_steps: int, method: str, target_product: str = None) -> pd.DataFrame:
    """
    Запускает циклическую симуляцию рынка векторно (NumPy-based).
    """
    sim_df = df.copy()
    
    if target_product and target_product != "Все товары":
        present_products = [target_product]
    else:
        present_products = [p for p in PRODUCTS if (sim_df['product'] == p).any()]
        
    if not present_products:
        return sim_df
    
    # Регрессия обучается только на переданной истории df
    prices_map = {}
    reg_params = {}
    if method == 'regression':
        for prod in present_products:
            a, b, opt_p, is_reliable = fit_regression(df, prod)
            reg_params[prod] = (a, b, is_reliable)
            if not is_reliable:
                last_price = float(df[df['product'] == prod].iloc[-1]['our_price'])
                opt_p = last_price
            prices_map[prod] = opt_p

    n_prods = len(present_products)
    base_prices = np.array([PRODUCTS[p]['base_price'] for p in present_products])
    base_sales = np.array([PRODUCTS[p]['base_sales'] for p in present_products])
    elasticities = np.array([PRODUCTS[p]['elasticity'] for p in present_products])
    cogs = np.array([PRODUCTS[p].get('cogs', 0.0) for p in present_products])
    product_ids = np.array([PRODUCTS[p]['id'] for p in present_products])

    reg_a = np.array([reg_params.get(p, (0.0, 0.0, False))[0] for p in present_products])
    reg_b = np.array([reg_params.get(p, (0.0, 0.0, False))[1] for p in present_products])
    reg_rel = np.array([reg_params.get(p, (0.0, 0.0, False))[2] for p in present_products])

    last_our_prices = np.zeros(n_prods)
    last_comp_prices = np.zeros(n_prods)
    last_sales = np.zeros(n_prods)
    sales_buffer = np.zeros((7, n_prods))

    for i, p in enumerate(present_products):
        hist = sim_df[sim_df['product'] == p]
        if not hist.empty:
            last_our_prices[i] = float(hist['our_price'].iloc[-1])
            last_comp_prices[i] = float(hist['competitor_price'].iloc[-1])
            last_sales[i] = float(hist['sales'].iloc[-1])
            hs_vals = hist['sales'].values
            if len(hs_vals) >= 7:
                sales_buffer[:, i] = hs_vals[-7:]
            else:
                sales_buffer[-len(hs_vals):, i] = hs_vals
                sales_buffer[:-len(hs_vals), i] = hs_vals.mean() if len(hs_vals) > 0 else 0

    rng = np.random.default_rng(SEED)
    last_date = sim_df['date'].max()
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)
    
    all_dates = np.repeat(date_range.values, n_prods)
    all_ids = np.tile(product_ids, n_steps)
    all_names = np.tile(present_products, n_steps)
    
    out_our_prices = np.zeros((n_steps, n_prods))
    out_comp_prices = np.zeros((n_steps, n_prods))
    out_sales = np.zeros((n_steps, n_prods))
    out_revenue = np.zeros((n_steps, n_prods))
    out_profit = np.zeros((n_steps, n_prods))
    out_cogs = np.tile(cogs, (n_steps, 1))

    for step in range(n_steps):
        rec_prices = np.zeros(n_prods)
        
        if method == 'rules':
            # Для правил нужно опираться на среднее за последние 7 дней
            avg7 = sales_buffer.mean(axis=0)
            for i, p in enumerate(present_products):
                row_mock = pd.Series({
                    'our_price': last_our_prices[i],
                    'competitor_price': last_comp_prices[i],
                    'sales': last_sales[i]
                })
                rp, _ = apply_rules(row_mock, avg7[i])
                rec_prices[i] = rp
        else:
            for i, p in enumerate(present_products):
                rec_prices[i] = prices_map.get(p, last_our_prices[i])

        comp_base = base_prices * 1.03
        noise_comp = rng.normal(0, 0.015 * base_prices, size=n_prods)
        competitor_prices = calc_competitor_prices(last_comp_prices, comp_base, last_our_prices, noise_comp)
        
        new_sales = np.zeros(n_prods)
        
        if method == 'regression':
            valid_mask = reg_rel & (reg_b > 0)
            
            # Надежная регрессия
            noise_scale_reg = np.maximum(2.0, 0.08 * np.maximum(np.abs(reg_a - reg_b * rec_prices), 1.0))
            new_sales[valid_mask] = calc_demand_regression(
                rec_prices, reg_a, reg_b, rng.normal(0, noise_scale_reg)
            )[valid_mask]
            
            # Откат на эвристику (ненадежная регрессия)
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                noise_rules = rng.normal(0, np.maximum(2.0, 0.08 * base_sales[invalid_mask]))
                new_sales[invalid_mask] = calc_demand_rules(
                    rec_prices[invalid_mask], competitor_prices[invalid_mask], base_prices[invalid_mask],
                    base_sales[invalid_mask], elasticities[invalid_mask], noise_rules
                )
        else:
            # Чистая эвристика
            noise = rng.normal(0, np.maximum(2.0, 0.08 * base_sales))
            new_sales = calc_demand_rules(
                rec_prices, competitor_prices, base_prices, base_sales, elasticities, noise
            )

        revenue = np.round(new_sales * rec_prices, 2)
        profit = np.round(revenue - new_sales * cogs, 2)
        
        # Обновляем state
        last_our_prices = rec_prices.copy()
        last_comp_prices = competitor_prices.copy()
        last_sales = new_sales.copy()
        
        sales_buffer[:-1, :] = sales_buffer[1:, :]
        sales_buffer[-1, :] = new_sales
        
        # Сохраняем в матрицы шага
        out_our_prices[step, :] = rec_prices
        out_comp_prices[step, :] = competitor_prices
        out_sales[step, :] = new_sales
        out_revenue[step, :] = revenue
        out_profit[step, :] = profit

    # Собираем результат воедино
    new_df = pd.DataFrame({
        'date': all_dates,
        'product_id': all_ids,
        'product': all_names,
        'our_price': out_our_prices.flatten(),
        'competitor_price': out_comp_prices.flatten(),
        'sales': out_sales.flatten(),
        'revenue': out_revenue.flatten(),
        'cogs': out_cogs.flatten(),
        'profit': out_profit.flatten(),
    })
    
    sim_df = pd.concat([sim_df, new_df], ignore_index=True)
    return sim_df
