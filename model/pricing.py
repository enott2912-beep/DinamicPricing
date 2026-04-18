import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ############################################################################
# 1. КОНСТАНТЫ И КОНФИГУРАЦИЯ (Единый источник правды)
# ############################################################################

SEED = 42

# Базовая информация о товарах: ID, стартовая цена, эластичность и базовый спрос
PRODUCTS = {
    'Молоко':   {'id': 1, 'base_price': 80,  'elasticity': 2.0, 'base_sales': 300},
    'Хлеб':     {'id': 2, 'base_price': 50,  'elasticity': 1.5, 'base_sales': 250},
    'Сок':      {'id': 3, 'base_price': 120, 'elasticity': 3.0, 'base_sales': 150},
    'Кофе':     {'id': 4, 'base_price': 450, 'elasticity': 1.2, 'base_sales': 80},
    'Шоколад':  {'id': 5, 'base_price': 100, 'elasticity': 2.5, 'base_sales': 200},
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


def _fit_linear_sales_vs_price(our_prices: np.ndarray, sales: np.ndarray, fallback_price: float) -> tuple[float, float, float, bool]:
    """
    Общая регрессия: Sales ≈ A - B*Price (B > 0 при надежной отрицательной эластичности).
    Оптимум выручки Rev = P*(A-B*P): P* = A/(2B).
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
        optimal_price = round(a / (2 * b), 2)
    else:
        optimal_price = float(fallback_price)
    return a, b, optimal_price, is_reliable


def fit_regression(df: pd.DataFrame, product: str) -> tuple[float, float, float, bool]:
    """
    Обучает линейную регрессию Sales = A - B * Price для одного товара (SKU).
    Вычисляет оптимальную цену через экстремум функции выручки Rev = P * (A - B*P).
    """
    prod_data = df[df['product'] == product]
    if len(prod_data) < 3:
        fallback = float(prod_data['our_price'].mean()) if len(prod_data) else 0.0
        return 0.0, 0.0, fallback, False
    return _fit_linear_sales_vs_price(
        prod_data['our_price'].values.astype(float),
        prod_data['sales'].values.astype(float),
        float(prod_data['our_price'].mean()),
    )


def fit_regression_aggregate_daily(work_df: pd.DataFrame) -> tuple[float, float, float, bool]:
    """
    Регрессия по дневному портфелю: одна точка на день — our_price (уже агрегат, напр. среднее), sales (сумма).
    Для графика «портфель»: та же математика, что и для SKU, на согласованных рядах.
    """
    if len(work_df) < 3:
        fb = float(work_df['our_price'].mean()) if len(work_df) else 0.0
        return 0.0, 0.0, fb, False
    return _fit_linear_sales_vs_price(
        work_df['our_price'].values.astype(float),
        work_df['sales'].values.astype(float),
        float(work_df['our_price'].mean()),
    )


def predict_sales_regression(a: float, b: float, price: float, noise: float = 0.0) -> int:
    """Спрос по линии регрессии (как в прогнозе рекомендаций), с опциональным шумом."""
    return max(0, int(round(a - b * float(price) + float(noise))))


def products_in_dataframe(df: pd.DataFrame) -> list[str]:
    """SKU из PRODUCTS, по которым есть строки в df (устойчивый порядок)."""
    return [p for p in PRODUCTS if not df.loc[df['product'] == p].empty]


def forecast(product: str, recommended_price: float, current_revenue: float) -> dict:
    """
    Прогноз выручки на основе 'идеальной' эластичности из конфигурации PRODUCTS.
    Только для одного SKU (имя из PRODUCTS).
    """
    p = PRODUCTS[product]
    price_dev = recommended_price - p['base_price']
    pred_sales = max(0, round(p['base_sales'] - p['elasticity'] * price_dev))
    pred_revenue = pred_sales * recommended_price
    
    growth_pct = ((pred_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0.0
    return {
        'forecast_sales': pred_sales,
        'forecast_revenue': round(pred_revenue, 2),
        'growth_pct': round(growth_pct, 1)
    }


def forecast_from_regression(a: float, b: float, recommended_price: float, current_revenue: float) -> dict:
    """
    Прогноз выручки на основе вычисленных параметров регрессии (реальные данные).
    """
    pred_sales = max(0, round(a - b * recommended_price))
    pred_revenue = pred_sales * recommended_price
    growth_pct = ((pred_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0.0
    return {
        'forecast_sales': pred_sales,
        'forecast_revenue': round(pred_revenue, 2),
        'growth_pct': round(growth_pct, 1)
    }


def predict_competitor_price(prev_comp_price: float, product_base_price: float, our_prev_price: float) -> float:
    """
    Рассчитывает ожидаемую цену конкурента на следующий период (шаг).
    Конкурент стремится к своей базовой цене, имеет инерцию (prev_comp_price) 
    и частично реагирует на нашу цену (our_prev_price).
    """
    comp_base = float(product_base_price * 1.03)
    competitor_price = (
        0.75 * float(prev_comp_price)
        + 0.20 * comp_base
        + 0.05 * float(our_prev_price)
        + np.random.normal(0, 0.015 * float(product_base_price))
    )
    return round(max(1.0, competitor_price), 2)


# ############################################################################
# 4. МОДУЛЬ СИМУЛЯЦИИ (TIME-ROLL FORWARD)
# ############################################################################


def simulate(df: pd.DataFrame, n_steps: int, method: str, target_product: str = None) -> pd.DataFrame:
    """
    Запускает циклическую симуляцию рынка (MVP Спринт 4).
    Алгоритм: оценка (на истории df) -> выбор цены -> генерация продаж -> повтор.

    При method='regression' регрессия Sales ≈ A - B·Price обучается **один раз** на
    переданном df (как на вкладке «Рекомендации»); спрос на шагах моделируется **той же**
    линией (плюс шум). При ненадежной регрессии спрос считается по формуле PRODUCTS.

    Аргументы:
        df: Исторические данные (на которых обучаются модели).
        n_steps: Горизонт прогнозирования (количество дней).
        method: 'rules' (эвристики) или 'regression' (математическая оптимизация).
        target_product: Продукт для симуляции (если None или «Все товары», симулируются все SKU из данных).

    Возвращает:
        pd.DataFrame: Набор данных, включающий историю и сгенерированные шаги.
    """
    np.random.seed(SEED)
    sim_df = df.copy()
    
    if target_product and target_product != "Все товары":
        present_products = [target_product]
    else:
        present_products = [p for p in PRODUCTS if (sim_df['product'] == p).any()]
    
    # Регрессия обучается только на переданной истории df (без синтетических шагов),
    # как на вкладке «Рекомендации» для выбранного периода.
    prices_map: dict[str, float] = {}
    reg_params: dict[str, tuple[float, float, bool]] = {}
    if method == 'regression':
        for prod in present_products:
            a, b, opt_p, is_reliable = fit_regression(df, prod)
            reg_params[prod] = (a, b, is_reliable)
            if not is_reliable:
                last_price = float(df[df['product'] == prod].iloc[-1]['our_price'])
                opt_p = last_price
            prices_map[prod] = opt_p

    # Основной цикл симуляции по дням
    for _ in range(n_steps):
        # В симуляции мы берем последнюю точку и генерируем следующую
        last_date = sim_df['date'].max()
        next_date = last_date + pd.Timedelta(days=1)
        
        new_rows = []
        for prod in present_products:
            p_info = PRODUCTS[prod]
            hist = sim_df[sim_df['product'] == prod]
            if hist.empty: continue
            last = hist.iloc[-1]
            
            # ВЫБОР ЦЕНЫ: по правилам или по регрессии
            if method == 'rules':
                avg7 = hist['sales'].tail(7).mean()
                rec_price, _ = apply_rules(last, avg7)
            else:
                rec_price = prices_map.get(prod, last['our_price'])
            
            # Конкурент тоже движется: возврат к своей базе + реакция на нашу цену + шум
            competitor_price = predict_competitor_price(last['competitor_price'], p_info['base_price'], rec_price)

            # ГЕНЕРАЦИЯ СПРОСА: при регрессии — та же линия Sales = A - B*P, что и в рекомендациях;
            # при правилах или ненадежной регрессии — формула из PRODUCTS (конкурент + шум).
            if method == 'regression':
                a, b, is_rel = reg_params.get(prod, (0.0, 0.0, False))
                if is_rel and b > 0:
                    mu_sales = a - b * float(rec_price)
                    noise_scale = max(2.0, 0.08 * max(abs(mu_sales), 1.0))
                    noise = float(np.random.normal(0, noise_scale))
                    new_sales = predict_sales_regression(a, b, rec_price, noise)
                else:
                    dev = rec_price - p_info['base_price']
                    comp_dev = rec_price - competitor_price
                    noise_scale = max(2.0, 0.08 * p_info['base_sales'])
                    noise = np.random.normal(0, noise_scale)
                    new_sales = max(
                        0,
                        round(p_info['base_sales'] - p_info['elasticity'] * dev - 0.5 * p_info['elasticity'] * comp_dev + noise),
                    )
            else:
                dev = rec_price - p_info['base_price']
                comp_dev = rec_price - competitor_price
                noise_scale = max(2.0, 0.08 * p_info['base_sales'])
                noise = np.random.normal(0, noise_scale)
                new_sales = max(
                    0,
                    round(p_info['base_sales'] - p_info['elasticity'] * dev - 0.5 * p_info['elasticity'] * comp_dev + noise),
                )
            
            new_rows.append({
                'date': next_date,
                'product_id': p_info['id'],
                'product': prod,
                'our_price': rec_price,
                'competitor_price': competitor_price,
                'sales': new_sales,
                'revenue': round(new_sales * rec_price, 2)
            })
        
        sim_df = pd.concat([sim_df, pd.DataFrame(new_rows)], ignore_index=True)
        
    return sim_df
