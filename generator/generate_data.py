"""
Генератор синтетических данных продаж.

Создаёт файл data/sales_history.csv с историей продаж за N_DAYS дней
для каждого товара из config.PRODUCTS.

Формула продаж:
    sales = base_sales - elasticity * (our_price - base_price) + noise
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Импорты из центральной модели
sys.path.append(str(Path(__file__).parent.parent))
from model.pricing import PRODUCTS, SEED
from model.math_engine import calc_competitor_prices, calc_demand_rules


def generate_all_data(n_days: int, start_date: datetime) -> pd.DataFrame:
    """Векторизованная генерация данных для всех товаров сразу."""
    rng = np.random.default_rng(SEED)
    
    product_names = list(PRODUCTS.keys())
    n_prods = len(product_names)
    
    base_prices = np.array([PRODUCTS[p]['base_price'] for p in product_names])
    base_sales = np.array([PRODUCTS[p]['base_sales'] for p in product_names])
    elasticities = np.array([PRODUCTS[p]['elasticity'] for p in product_names])
    product_ids = np.array([PRODUCTS[p]['id'] for p in product_names])
    cogs = np.array([PRODUCTS[p].get('cogs', 0.0) for p in product_names])
    
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Цены
    base_price_mat = np.tile(base_prices, (n_days, 1))
    our_prices = np.round(base_price_mat * rng.uniform(0.9, 1.1, size=(n_days, n_prods)), 2)
    
    comp_base = base_prices * 1.03
    competitor_prices = np.zeros((n_days, n_prods))
    competitor_prices[0] = np.maximum(1, comp_base + rng.normal(0, 0.02 * base_prices, size=n_prods))
    
    for i in range(1, n_days):
        noise = rng.normal(0, 0.015 * base_prices, size=n_prods)
        competitor_prices[i] = calc_competitor_prices(competitor_prices[i - 1], comp_base, our_prices[i - 1], noise)
        
    noise_scale = np.maximum(2.0, 0.08 * base_sales)
    noise_mat = np.tile(noise_scale, (n_days, 1))
    noise = rng.normal(0, noise_mat, size=(n_days, n_prods))
    
    elast_mat = np.tile(elasticities, (n_days, 1))
    base_sales_mat = np.tile(base_sales, (n_days, 1))
    
    sales = calc_demand_rules(
        our_prices=our_prices,
        competitor_prices=competitor_prices,
        base_prices=base_price_mat,
        base_sales=base_sales_mat,
        elasticities=elast_mat,
        noise=noise
    )
    
    revenue = np.round(sales * our_prices, 2)
    cogs_mat = np.tile(cogs, (n_days, 1))
    profit = np.round(revenue - sales * cogs_mat, 2)
    
    date_col = np.repeat(dates, n_prods)
    product_id_col = np.tile(product_ids, n_days)
    product_col = np.tile(product_names, n_days)
    cogs_col = np.tile(cogs, n_days)
    
    df = pd.DataFrame({
        'date': date_col,
        'product_id': product_id_col,
        'product': product_col,
        'our_price': our_prices.flatten(),
        'competitor_price': competitor_prices.flatten(),
        'sales': sales.flatten(),
        'revenue': revenue.flatten(),
        'cogs': cogs_col,
        'profit': profit.flatten()
    })
    
    return df.sort_values(['date', 'product']).reset_index(drop=True)


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет DataFrame в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Данные сохранены: {path.absolute()}")


def main(n_days: int = 100) -> None:
    """Точка входа."""
    np.random.seed(SEED)
    n_days = int(max(1, n_days))
    
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'sales_history.csv'
    predict_path = base_dir / 'data' / 'predict_sales.csv'

    # Каждый запуск генератора полностью перезаписывает историю.
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days - 1)
    print(f"Полная перегенерация истории за {n_days} дней...")
    df = generate_all_data(n_days, start_date)

    save_data(df, data_path)
    # Прогнозы должны храниться отдельно и очищаться при новой генерации истории.
    pd.DataFrame(columns=df.columns).to_csv(predict_path, index=False)
    
    print("\nПример данных:")
    print(df.head())
    print("\nКорреляция Цена-Продажи (должна быть < 0):")
    for p in PRODUCTS:
        sub = df[df['product'] == p]
        corr = sub['our_price'].corr(sub['sales'])
        print(f" - {p}: {corr:.4f}")

    print("\nГенерация завершена успешно.")


if __name__ == '__main__':
    main()
