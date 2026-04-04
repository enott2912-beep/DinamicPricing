"""
Рекомендатор цен (Rule-based).

Загружает данные из data/sales_history.csv, применяет три правила к последнему дню
каждого товара и сохраняет рекомендации в data/recommendations.csv.

Правила:
  1. Конкурент дешевле на 10%+ → снижаем на 5%
  2. Продажи ниже нормы на 20%+ → снижаем на 1₽
  3. Иначе → держим цену
"""

import pandas as pd
from pathlib import Path

from config import PRODUCTS


def load_data(filepath: Path) -> pd.DataFrame:
    """Загружает и парсит CSV с датами."""
    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_last_day(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Возвращает данные за последний день и средние продажи за 7 дней по товарам."""
    last_date = df['date'].max()
    last_day = df[df['date'] == last_date].copy()

    seven_days_ago = last_date - pd.Timedelta(days=6)
    last_7d = df[(df['date'] >= seven_days_ago) & (df['date'] <= last_date)]
    avg_sales = last_7d.groupby('product')['sales'].mean().to_dict()

    return last_day, avg_sales


def apply_rules(row: pd.Series, avg_sales: float) -> tuple[float, str]:
    """Применяет правила ценообразования к одной записи."""
    our_price = row['our_price']

    if row['competitor_price'] < our_price * 0.90:
        return round(our_price * 0.95, 2), "competitor_undercut"
    if row['sales'] < avg_sales * 0.80:
        return round(our_price - 1.0, 2), "low_sales"
    return our_price, "hold"


def forecast_revenue(product: str, rec_price: float,
                     current_sales: float, current_price: float) -> dict:
    """Рассчитывает прогноз продаж и изменение выручки при новой цене."""
    params = PRODUCTS[product]
    price_dev = rec_price - params['base_price']
    pred_sales = max(0, round(params['base_sales'] - params['elasticity'] * price_dev))
    pred_revenue = pred_sales * rec_price
    cur_revenue = current_sales * current_price

    delta_pct = ((pred_revenue - cur_revenue) / cur_revenue * 100) if cur_revenue > 0 else 0.0

    return {
        'predicted_sales': pred_sales,
        'predicted_revenue': round(pred_revenue, 2),
        'revenue_change_pct': round(delta_pct, 1),
    }


def main() -> None:
    """Точка входа: загрузка → правила → прогноз → сохранение."""
    base_dir = Path(__file__).parent
    data_path = base_dir / 'data' / 'sales_history.csv'
    out_path = base_dir / 'data' / 'recommendations.csv'

    df = load_data(data_path)
    last_day, avg_sales_dict = get_last_day(df)
    results = []

    for _, row in last_day.iterrows():
        product = row['product']
        if row['our_price'] == 0 or pd.isna(row['our_price']):
            continue

        avg = avg_sales_dict.get(product, row['sales'])
        rec_price, rule = apply_rules(row, avg)
        fc = forecast_revenue(product, rec_price, row['sales'], row['our_price'])

        sign = "+" if fc['revenue_change_pct'] > 0 else ""
        print(f"{product}: {row['our_price']:.2f} → {rec_price:.2f} ₽ "
              f"| {rule} | {sign}{fc['revenue_change_pct']}%")

        results.append({
            'product': product,
            'our_price': row['our_price'],
            'recommended_price': rec_price,
            'rule_applied': rule,
            **fc,
        })

    if results:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\nСохранено: {out_path.absolute()}")


if __name__ == '__main__':
    main()
