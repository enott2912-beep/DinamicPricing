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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from config import PRODUCTS, N_DAYS, SEED


def generate_product_data(product: str, n_days: int) -> pd.DataFrame:
    """Генерирует данные продаж для одного товара за n_days дней."""
    product_hash = sum(ord(c) for c in product)
    np.random.seed(SEED + product_hash % 10000)

    params = PRODUCTS[product]
    base_price = params['base_price']
    elasticity = params['elasticity']
    base_sales = params['base_sales']

    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    our_prices = np.round(base_price * np.random.uniform(0.9, 1.1, size=n_days), 2)
    competitor_prices = np.round(our_prices * np.random.uniform(0.85, 1.15, size=n_days), 2)

    price_deviation = our_prices - base_price
    noise = np.random.normal(0, 5, size=n_days)
    sales = np.maximum(0, np.round(base_sales - elasticity * price_deviation + noise))
    revenue = sales * our_prices

    return pd.DataFrame({
        'date': dates,
        'product_id': params['id'],
        'product': product,
        'our_price': our_prices,
        'competitor_price': competitor_prices,
        'sales': sales,
        'revenue': revenue,
    })


def generate_all_data() -> pd.DataFrame:
    """Объединяет данные всех товаров в один DataFrame."""
    frames = [generate_product_data(p, N_DAYS) for p in PRODUCTS]
    return pd.concat(frames, ignore_index=True).sort_values(['date', 'product']).reset_index(drop=True)


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет DataFrame в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Данные сохранены: {path.absolute()}")


def plot_data(df: pd.DataFrame) -> None:
    """Строит и сохраняет графики: scatter (цена vs продажи), line (выручка по дням)."""
    plots_dir = Path(__file__).parent.parent / 'data' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: our_price vs sales
    fig, ax = plt.subplots(figsize=(10, 6))
    for product in df['product'].unique():
        sub = df[df['product'] == product]
        ax.scatter(sub['our_price'], sub['sales'], label=product, alpha=0.7)
    ax.set(title='Зависимость продаж от цены', xlabel='Наша цена', ylabel='Продажи')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.savefig(plots_dir / 'price_vs_sales_scatter.png', bbox_inches='tight')
    plt.close(fig)

    # Line: revenue по дням
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_rev = df.groupby('date')['revenue'].sum()
    ax.plot(daily_rev.index, daily_rev.values, marker='o', linestyle='-', color='#1f77b4')
    ax.set(title='Суммарная выручка по дням', xlabel='Дата', ylabel='Выручка')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    fig.savefig(plots_dir / 'daily_revenue_line.png', bbox_inches='tight')
    plt.close(fig)

    print(f"Графики сохранены: {plots_dir.absolute()}")


def main() -> None:
    """Точка входа: генерация данных, вывод статистики, сохранение и графики."""
    np.random.seed(SEED)
    print("Генерация данных...")

    df = generate_all_data()
    data_path = Path(__file__).parent.parent / 'data' / 'sales_history.csv'

    print(f"\nShape: {df.shape}")
    print(f"\n{df.head()}")
    print(f"\n{df.describe()}")

    save_data(df, data_path)
    plot_data(df)
    print("Готово.")


if __name__ == '__main__':
    main()
