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

# Импорты из центральной модели
sys.path.append(str(Path(__file__).parent.parent))
from model.pricing import PRODUCTS, SEED


def generate_product_data(product: str, n_days: int, start_date: datetime) -> pd.DataFrame:
    """Генерирует данные продаж для одного товара за n_days дней."""
    # Фиксируем воспроизводимость для каждого товара
    product_hash = sum(ord(c) for c in product)
    np.random.seed(SEED + product_hash % 10000)

    params = PRODUCTS[product]
    base_price = params['base_price']
    elasticity = params['elasticity']
    base_sales = params['base_sales']

    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')

    # Цены колеблются вокруг базы
    our_prices = np.round(base_price * np.random.uniform(0.9, 1.1, size=n_days), 2)
    competitor_base = base_price * 1.03
    competitor_prices = np.zeros(n_days)
    competitor_prices[0] = max(
        1,
        competitor_base + np.random.normal(0, 0.02 * base_price)
    )
    for i in range(1, n_days):
        # Конкурент имеет свою базу, не следует механически за нашей ценой.
        competitor_prices[i] = (
            0.75 * competitor_prices[i - 1]
            + 0.20 * competitor_base
            + 0.05 * our_prices[i - 1]
            + np.random.normal(0, 0.015 * base_price)
        )
    competitor_prices = np.round(np.maximum(1, competitor_prices), 2)

    # Формула: Учитываем разницу с базой и разницу с конкурентом
    # (our - comp) > 0 -> спрос падает
    price_diff_base = our_prices - base_price
    price_diff_comp = our_prices - competitor_prices
    
    noise_scale = max(2.0, 0.08 * base_sales)
    noise = np.random.normal(0, noise_scale, size=n_days)
    # Основная эластичность + влияние конкурента (вес 0.5)
    sales = np.maximum(0, np.round(
        base_sales - elasticity * price_diff_base - 0.5 * elasticity * price_diff_comp + noise
    ))
    
    revenue = np.round(sales * our_prices, 2)

    return pd.DataFrame({
        'date': dates,
        'product_id': params['id'],
        'product': product,
        'our_price': our_prices,
        'competitor_price': competitor_prices,
        'sales': sales,
        'revenue': revenue,
    })


def generate_all_data(n_days: int, start_date: datetime) -> pd.DataFrame:
    """Объединяет данные всех товаров в один DataFrame."""
    frames = [generate_product_data(p, n_days, start_date) for p in PRODUCTS]
    return pd.concat(frames, ignore_index=True).sort_values(['date', 'product']).reset_index(drop=True)


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет DataFrame в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Данные сохранены: {path.absolute()}")


def plot_data(df: pd.DataFrame) -> None:
    """Строит и сохраняет графики в папку data/plots."""
    plots_dir = Path(__file__).parent.parent / 'data' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: Наша цена vs Продажи
    fig, ax = plt.subplots(figsize=(10, 6))
    for product in df['product'].unique():
        sub = df[df['product'] == product]
        ax.scatter(sub['our_price'], sub['sales'], label=product, alpha=0.7)
    ax.set(title='Зависимость продаж от цены (с учетом демпфирования конкурентов)', 
           xlabel='Цена (₽)', ylabel='Продажи (шт)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.savefig(plots_dir / 'price_vs_sales_scatter.png', bbox_inches='tight')
    plt.close(fig)

    # Линейный: Выручка по дням
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    revenue_daily = df.groupby('date')['revenue'].sum()
    ax2.plot(revenue_daily.index, revenue_daily.values, color='green', marker='o', linestyle='-', alpha=0.8)
    ax2.set(title='Общая ежедневная выручка (все товары)', 
            xlabel='Дата', ylabel='Выручка (₽)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    fig2.savefig(plots_dir / 'daily_revenue_line.png', bbox_inches='tight')
    plt.close(fig2)

    print(f"Графики сохранены в {plots_dir}")


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
    plot_data(df)
    
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
