import numpy as np

def calc_competitor_prices(
    last_comp_prices: np.ndarray, 
    comp_base_prices: np.ndarray, 
    our_prices: np.ndarray, 
    noise: np.ndarray
) -> np.ndarray:
    """
    Векторизованный пересчет цен конкурента с учетом инерции и нашей цены.
    Реализует принцип DRY: используется в генераторе и в симуляторе.
    """
    new_comp_prices = 0.75 * last_comp_prices + 0.20 * comp_base_prices + 0.05 * our_prices + noise
    return np.round(np.maximum(1.0, new_comp_prices), 2)

def calc_demand_rules(
    our_prices: np.ndarray, 
    competitor_prices: np.ndarray, 
    base_prices: np.ndarray, 
    base_sales: np.ndarray, 
    elasticities: np.ndarray, 
    noise: np.ndarray
) -> np.ndarray:
    """
    Векторизованный расчет спроса (продаж) по базовой эластичности (эвристика/правила).
    Спрос падает от разницы с нашей базой и от разницы с конкурентом.
    """
    dev = our_prices - base_prices
    comp_dev = our_prices - competitor_prices
    sales = base_sales - elasticities * dev - 0.5 * elasticities * comp_dev + noise
    return np.maximum(0, np.round(sales))

def calc_demand_regression(
    our_prices: np.ndarray, 
    reg_a: np.ndarray, 
    reg_b: np.ndarray, 
    noise: np.ndarray
) -> np.ndarray:
    """
    Векторизованный расчет спроса (продаж) по предсчитанному наклону регрессии.
    """
    mu_sales = reg_a - reg_b * our_prices
    return np.maximum(0, np.round(mu_sales + noise))
