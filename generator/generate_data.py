import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Константы
PRODUCTS = {
    'Молоко': {'base_price': 80, 'elasticity': 2.0, 'base_sales': 300},
    'Хлеб': {'base_price': 50, 'elasticity': 1.5, 'base_sales': 250},
    'Сок': {'base_price': 120, 'elasticity': 3.0, 'base_sales': 150},
    'Кофе': {'base_price': 450, 'elasticity': 1.2, 'base_sales': 80},
    'Шоколад': {'base_price': 100, 'elasticity': 2.5, 'base_sales': 200}
}
N_DAYS = 100
SEED = 42

def generate_product_data(product: str, n_days: int) -> pd.DataFrame:
    """
    Генерирует исторические данные продаж для одного товара.
    
    Параметры:
        product (str): Название товара.
        n_days (int): Количество дней для генерации.
        
    Возвращает:
        pd.DataFrame: DataFrame с данными по одному товару.
    """
    product_hash = sum(ord(c) for c in product) # Стабильный seed вместо hash()
    np.random.seed(SEED + product_hash % 10000) # Уникальный seed для добавления шума каждому товару, сохраняя воспроизводимость
    
    params = PRODUCTS[product]
    base_price = params['base_price']
    elasticity = params['elasticity']
    base_sales = params['base_sales']
    
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Генерация цен
    our_prices = base_price * np.random.uniform(0.9, 1.1, size=n_days)
    our_prices = np.round(our_prices, 2)
    
    # Цены конкурентов (отклонение +- 15% от нашей цены)
    competitor_prices = our_prices * np.random.uniform(0.85, 1.15, size=n_days)
    competitor_prices = np.round(competitor_prices, 2)
    
    # Расчет продаж (зависит от отклонения цены от базовой)
    random_noise = np.random.normal(0, 5, size=n_days)
    price_deviation = our_prices - base_price
    sales = base_sales - elasticity * price_deviation + random_noise
    
    # Защита от отрицательных значений
    sales = np.maximum(0, np.round(sales))
    
    # Выручка
    revenue = sales * our_prices
    
    df = pd.DataFrame({
        'date': dates,
        'product': product,
        'our_price': our_prices,
        'competitor_price': competitor_prices,
        'sales': sales,
        'revenue': revenue
    })
    
    return df

def generate_all_data() -> pd.DataFrame:
    """
    Объединяет данные всех товаров в один DataFrame.
    
    Возвращает:
        pd.DataFrame: Общий датасет продаж.
    """
    all_data = []
    for product in PRODUCTS.keys():
        df = generate_product_data(product, N_DAYS)
        all_data.append(df)
        
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(by=['date', 'product']).reset_index(drop=True)
    return final_df

def save_data(df: pd.DataFrame, path: Path):
    """
    Сохраняет DataFrame в CSV файл.
    
    Параметры:
        df (pd.DataFrame): Датасет для сохранения.
        path (Path): Путь к итоговому файлу (запись).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nДанные успешно сохранены в: {path.absolute()}")

def plot_data(df: pd.DataFrame):
    """
    Создает и сохраняет визуализации данных. Распределяет графики по товарам.
    
    Параметры:
        df (pd.DataFrame): Исходные данные для графиков.
    """
    # Папка для графиков
    plots_dir = Path(__file__).parent.parent / 'data' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # График 1: scatter - our_price vs sales
    plt.figure(figsize=(10, 6))
    for product in df['product'].unique():
        prod_data = df[df['product'] == product]
        plt.scatter(prod_data['our_price'], prod_data['sales'], label=product, alpha=0.7)
        
    plt.title('Зависимость продаж от цены (our_price vs sales)')
    plt.xlabel('Наша цена')
    plt.ylabel('Продажи')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    scatter_path = plots_dir / 'price_vs_sales_scatter.png'
    plt.savefig(scatter_path, bbox_inches='tight')
    plt.close()
    print(f"График (scatter) сохранен в: {scatter_path.absolute()}")
    
    # График 2: line - revenue по дням
    plt.figure(figsize=(12, 6))
    daily_revenue = df.groupby('date')['revenue'].sum().reset_index()
    plt.plot(daily_revenue['date'], daily_revenue['revenue'], marker='o', linestyle='-', color='#1f77b4')
    
    plt.title('Суммарная выручка по дням')
    plt.xlabel('Дата')
    plt.ylabel('Выручка')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    
    line_path = plots_dir / 'daily_revenue_line.png'
    plt.savefig(line_path, bbox_inches='tight')
    plt.close()
    print(f"График (line) сохранен в: {line_path.absolute()}")

def main():
    """
    Основная логика скрипта:
    Запуск параметров, генерация данных, вывод метрик и перенаправление на сохранение и графики.
    """
    # Фиксация seed для общей воспроизводимости, если потребуется np.random напрямую
    np.random.seed(SEED)
    
    print("Генерация данных начата...")
    df = generate_all_data()
    
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'sales_history.csv'
    
    print(f"\n--- ИНФОРМАЦИЯ О ДАННЫХ ---")
    print(f"Shape датасета: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nОписательные статистики (describe):")
    print(df.describe())
    print("---------------------------\n")
    
    save_data(df, data_path)
    plot_data(df)
    
    print("Скрипт успешно завершен.")

if __name__ == '__main__':
    main()
