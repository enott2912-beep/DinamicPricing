import pandas as pd
from pathlib import Path

from config import PRODUCTS

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Загружает данные о продажах из указанного CSV файла.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Файл {filepath} не найден.")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_last_day(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Возвращает данные за последний день для каждого товара
    и словарь со средними продажами за последние 7 дней.
    """
    last_date = df['date'].max()
    last_day_df = df[df['date'] == last_date].copy()
    
    # Расчет средних продаж за последние 7 дней (включая последний день)
    seven_days_ago = last_date - pd.Timedelta(days=6)
    last_7_days = df[(df['date'] >= seven_days_ago) & (df['date'] <= last_date)]
    avg_sales_dict = last_7_days.groupby('product')['sales'].mean().to_dict()
    
    return last_day_df, avg_sales_dict

def apply_rules(row: pd.Series, avg_sales: float) -> tuple[float, str]:
    """
    Применяет бизнес-правила для формирования рекомендованной цены.
    Правила применяются последовательно.
    """
    our_price = row['our_price']
    competitor_price = row['competitor_price']
    sales = row['sales']
    
    # ПРАВИЛО 1 — конкурент сильно дешевле
    if competitor_price < our_price * 0.90:
        recommended_price = round(our_price * 0.95, 2)
        rule_applied = "competitor_undercut"
    # ПРАВИЛО 2 — продажи ниже нормы
    elif sales < avg_sales * 0.80:
        recommended_price = round(our_price - 1.0, 2)
        rule_applied = "low_sales"
    # ПРАВИЛО 3 — всё в норме
    else:
        recommended_price = our_price
        rule_applied = "hold"
        
    return recommended_price, rule_applied

def forecast_revenue(product: str, recommended_price: float, current_sales: float, current_price: float) -> dict:
    """
    Рассчитывает прогноз продаж и выручки по формуле эластичности,
    а также изменение выручки в процентах.
    """
    params = PRODUCTS[product]
    base_price = params['base_price']
    elasticity = params['elasticity']
    base_sales = params['base_sales']
    
    price_deviation_new = recommended_price - base_price
    predicted_sales = base_sales - elasticity * price_deviation_new
    predicted_sales = max(0, round(predicted_sales))
    
    predicted_revenue = predicted_sales * recommended_price
    current_revenue = current_sales * current_price
    
    if current_revenue > 0:
        revenue_change_pct = (predicted_revenue - current_revenue) / current_revenue * 100.0
    else:
        revenue_change_pct = 0.0
        
    return {
        'predicted_sales': predicted_sales,
        'predicted_revenue': round(predicted_revenue, 2),
        'revenue_change_pct': round(revenue_change_pct, 1)
    }

def main():
    """
    Основная логика скрипта рекомендаций: 
    загрузка, применение правил, прогноз, вывод и сохранение.
    """
    base_dir = Path(__file__).parent
    data_path = base_dir / 'data' / 'sales_history.csv'
    out_path = base_dir / 'data' / 'recommendations.csv'
    
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return
        
    last_day_df, avg_sales_dict = get_last_day(df)
    results = []
    
    for _, row in last_day_df.iterrows():
        product = row['product']
        our_price = row['our_price']
        competitor_price = row['competitor_price']
        sales = row['sales']
        
        # Защита от нулевой цены и отсутствующих данных
        if our_price == 0 or pd.isna(our_price) or pd.isna(sales) or pd.isna(competitor_price):
            print(f"Внимание: пропущен товар '{product}' (отсутствуют данные или нулевая цена).")
            continue
            
        avg_sales = avg_sales_dict.get(product, sales)
        
        recommended_price, rule_applied = apply_rules(row, avg_sales)
        forecast = forecast_revenue(product, recommended_price, sales, our_price)
        
        change_sign = "+" if forecast['revenue_change_pct'] > 0 else ""
        print(f"{product}: {our_price:.2f} руб → {recommended_price:.2f} руб | правило: {rule_applied} | прогноз выручки: {change_sign}{forecast['revenue_change_pct']}%")
        
        results.append({
            'product': product,
            'our_price': our_price,
            'competitor_price': competitor_price,
            'sales': sales,
            'recommended_price': recommended_price,
            'rule_applied': rule_applied,
            'predicted_sales': forecast['predicted_sales'],
            'predicted_revenue': forecast['predicted_revenue'],
            'revenue_change_pct': forecast['revenue_change_pct']
        })
        
    if results:
        results_df = pd.DataFrame(results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"\nРекомендации успешно сохранены в: {out_path.absolute()}")
    else:
        print("\nНет данных для сохранения.")

if __name__ == '__main__':
    main()
