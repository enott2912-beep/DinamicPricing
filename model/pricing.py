import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ==========================================
# КОНСТАНТЫ (Единственный источник правды)
# ==========================================
SEED = 42

PRODUCTS = {
    'Молоко':   {'id': 1, 'base_price': 80,  'elasticity': 2.0, 'base_sales': 300},
    'Хлеб':     {'id': 2, 'base_price': 50,  'elasticity': 1.5, 'base_sales': 250},
    'Сок':      {'id': 3, 'base_price': 120, 'elasticity': 3.0, 'base_sales': 150},
    'Кофе':     {'id': 4, 'base_price': 450, 'elasticity': 1.2, 'base_sales': 80},
    'Шоколад':  {'id': 5, 'base_price': 100, 'elasticity': 2.5, 'base_sales': 200},
}

# ==========================================
# ЛОГИКА ЦЕНООБРАЗОВАНИЯ (БИЗНЕС-ПРАВИЛА)
# ==========================================

def apply_rules(row: pd.Series, avg_sales_7d: float) -> tuple[float, str]:
    """
    Эвристические правила (rule-based).
    Возвращает (рекомендованная_цена, название_правила).
    """
    price = row['our_price']
    
    # Правило 1: Реакция на конкурента (Спринт 2)
    # Если конкурент значительно дешевле (на 10% и более), снижаем цену на 5%
    if row['competitor_price'] < price * 0.90:
        return round(price * 0.95, 2), "competitor_undercut"
    
    # Правило 2: Реакция на падение спроса (Спринт 2)
    # Если продажи за вчера на 20% ниже скользящего среднего, пробуем стимулировать спрос
    if row['sales'] < avg_sales_7d * 0.80:
        return round(price - 1.0, 2), "low_sales"
    
    return price, "hold"


def fit_regression(df: pd.DataFrame, product: str) -> tuple[float, float, float, bool]:
    """
    Обучает регрессию Sales = A - B * Price.
    Возвращает (a, b, optimal_price, is_reliable).
    """
    prod_data = df[df['product'] == product]
    if len(prod_data) < 3:
        fallback = float(prod_data['our_price'].mean()) if len(prod_data) else 0.0
        return 0.0, 0.0, fallback, False
        
    X = prod_data['our_price'].values.reshape(-1, 1)
    y = prod_data['sales'].values
    
    model = LinearRegression().fit(X, y)
    a = model.intercept_
    coef = float(model.coef_[0])
    is_reliable = coef < 0
    b = -coef if coef < 0 else 0.0
    
    # МАТЕМАТИЧЕСКАЯ МОДЕЛЬ (Спринт 3):
    # Выручка (Revenue) = P * (A - B*P) = AP - BP^2
    # Для максимизации ищем производную dRev/dP:
    # dRev/dP = A - 2BP. Приравниваем к нулю: A - 2BP = 0 -> P = A / 2B
    optimal_price = round(a / (2 * b), 2) if b > 0 else float(prod_data['our_price'].mean())
    
    return float(a), float(b), float(optimal_price), is_reliable


def forecast(product: str, recommended_price: float, current_revenue: float) -> dict:
    """
    Прогноз выручки на основе эластичности из конфига.
    """
    p = PRODUCTS[product]
    # Формула с учетом эластичности
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
    Прогноз выручки из параметров регрессии Sales = A - B*Price.
    """
    pred_sales = max(0, round(a - b * recommended_price))
    pred_revenue = pred_sales * recommended_price
    growth_pct = ((pred_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0.0
    return {
        'forecast_sales': pred_sales,
        'forecast_revenue': round(pred_revenue, 2),
        'growth_pct': round(growth_pct, 1)
    }


def simulate(df: pd.DataFrame, n_steps: int, method: str) -> pd.DataFrame:
    """
    Симуляция рынка на n_steps вперед.
    method: 'rules' или 'regression'
    """
    np.random.seed(SEED)
    sim_df = df.copy()
    
    present_products = [p for p in PRODUCTS if (sim_df['product'] == p).any()]
    retrain_every = 3

    # Предрасчет цен для регрессии
    prices_map = {}
    if method == 'regression':
        for prod in present_products:
            _, _, opt_p, is_reliable = fit_regression(sim_df, prod)
            if not is_reliable:
                last_price = float(sim_df[sim_df['product'] == prod].iloc[-1]['our_price'])
                opt_p = last_price
            prices_map[prod] = opt_p

    for step in range(n_steps):
        if method == 'regression' and step % retrain_every == 0:
            for prod in present_products:
                _, _, opt_p, is_reliable = fit_regression(sim_df, prod)
                if not is_reliable:
                    last_price = float(sim_df[sim_df['product'] == prod].iloc[-1]['our_price'])
                    opt_p = last_price
                prices_map[prod] = opt_p

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
            prev_comp = float(last['competitor_price'])
            comp_base = float(p_info['base_price'] * 1.03)
            competitor_price = (
                0.70 * prev_comp
                + 0.20 * comp_base
                + 0.10 * float(rec_price)
                + np.random.normal(0, 0.015 * p_info['base_price'])
            )
            competitor_price = round(max(1, competitor_price), 2)

            # ГЕНЕРАЦИЯ НОВОГО СПРОСА (Симуляция Спринта 4)
            # Учитываем эластичность к нашей цене и разницу с конкурентом
            dev = rec_price - p_info['base_price']
            comp_dev = rec_price - competitor_price
            
            noise_scale = max(2.0, 0.08 * p_info['base_sales'])
            noise = np.random.normal(0, noise_scale) # Масштабируем шум относительно спроса
            
            # Формула: Базовый спрос - (эластичность * отклонение от базы) - (влияние конкурента)
            new_sales = max(0, round(p_info['base_sales'] - p_info['elasticity']*dev - 0.5*p_info['elasticity']*comp_dev + noise))
            
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
