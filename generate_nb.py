import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

add_md("# 0. Импорты и константы")
add_code('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
from IPython.display import display

%matplotlib inline

SEED = 42
np.random.seed(SEED)

PRODUCTS = {
    'Молоко':   {'base_price': 80,  'elasticity': 2.0, 'base_sales': 300},
    'Хлеб':    {'base_price': 50,  'elasticity': 1.5, 'base_sales': 250},
    'Сок':     {'base_price': 120, 'elasticity': 3.0, 'base_sales': 150},
    'Кофе':    {'base_price': 450, 'elasticity': 1.2, 'base_sales': 80},
    'Шоколад': {'base_price': 100, 'elasticity': 2.5, 'base_sales': 200}
}

# --- ФУНКЦИИ ---
def apply_rules(row: pd.Series, avg_sales_7d: float) -> tuple[float, str]:
    """Применяет эвристические правила к последнему дню для определения рекомендуемой цены."""
    our_price = row['our_price']
    competitor_price = row['competitor_price']
    sales = row['sales']
    
    if competitor_price < our_price * 0.90:
        return round(our_price * 0.95, 2), "competitor_undercut"
    elif sales < avg_sales_7d * 0.80:
        return round(our_price - 1.0, 2), "low_sales"
    else:
        return our_price, "hold"

def forecast(product: str, rec_price: float, current_revenue: float) -> tuple[int, float, float]:
    """Прогноз объема продаж и выручки после изменения цены."""
    params = PRODUCTS[product]
    base_price = params['base_price']
    elasticity = params['elasticity']
    base_sales = params['base_sales']
    
    price_dev = rec_price - base_price
    predicted_sales = max(0, round(base_sales - elasticity * price_dev))
    predicted_revenue = predicted_sales * rec_price
    
    delta_pct = 0.0
    if current_revenue > 0:
        delta_pct = (predicted_revenue - current_revenue) / current_revenue * 100
        
    return predicted_sales, predicted_revenue, delta_pct

def simulate(df_history: pd.DataFrame, n_steps: int = 10, method: str = 'regression') -> pd.DataFrame:
    """Симуляция поведения рынка на n_steps дней вперед."""
    sim_df = df_history.copy()
    
    # Предподготовка для регрессии
    reg_results = {}
    if method == 'regression':
        for prod in PRODUCTS:
            sub = sim_df[sim_df['product'] == prod]
            if not sub.empty:
                X = sub['our_price'].values.reshape(-1, 1)
                y = sub['sales'].values
                model = LinearRegression().fit(X, y)
                a = model.intercept_
                b = abs(model.coef_[0])
                optimal_price = round(a / (2 * b), 2)
                reg_results[prod] = optimal_price

    for step in range(n_steps):
        last_date = pd.to_datetime(sim_df['date']).max()
        next_date = last_date + pd.Timedelta(days=1)
        next_date_str = next_date.strftime('%Y-%m-%d')
        
        new_rows = []
        for prod in PRODUCTS:
            prod_hist = sim_df[sim_df['product'] == prod]
            if prod_hist.empty:
                continue
            last_row = prod_hist.iloc[-1]
            
            if method == 'rules':
                avg_sales_7d = prod_hist['sales'].tail(7).mean()
                rec_price, _ = apply_rules(last_row, avg_sales_7d)
            else:
                rec_price = reg_results.get(prod, last_row['our_price'])
                
            params = PRODUCTS[prod]
            price_dev = rec_price - params['base_price']
            new_sales = params['base_sales'] - params['elasticity'] * price_dev + np.random.normal(0, 5)
            new_sales = max(0, round(new_sales))
            new_revenue = new_sales * rec_price
            
            new_row = {
                'date': next_date_str,
                'product_id': last_row.get('product_id', 0),
                'product': prod,
                'our_price': rec_price,
                'competitor_price': last_row['competitor_price'], # оставляем без изменений
                'sales': new_sales,
                'revenue': new_revenue
            }
            new_rows.append(new_row)
            
        sim_df = pd.concat([sim_df, pd.DataFrame(new_rows)], ignore_index=True)
            
    return sim_df
''')

add_md("# 1. Загрузка и обзор данных (EDA)")
add_code('''
# Загрузить data/sales_history.csv через pandas
data_path = Path('data/sales_history.csv')
if not data_path.exists():
    data_path = Path('../data/sales_history.csv')
    
df = pd.read_csv(data_path)

# Вывести: shape, dtypes, head(5), describe()
print("Shape:", df.shape)
print("\\nDtypes:\\n", df.dtypes)
print("\\nDescribe:\\n", df.describe())
display(df.head(5))

# График 1 — scatter: our_price vs sales, цвет = товар
plt.figure(figsize=(10, 5))
for product in df['product'].unique():
    prod_data = df[df['product'] == product]
    plt.scatter(prod_data['our_price'], prod_data['sales'], label=product)
plt.title('Зависимость продаж от цены')
plt.xlabel('our_price')
plt.ylabel('sales')
plt.grid(True)
plt.legend()
plt.show()

# График 2 — line: суммарная выручка по дням
plt.figure(figsize=(10, 5))
revenue_by_date = df.groupby('date')['revenue'].sum()
revenue_by_date.index = pd.to_datetime(revenue_by_date.index)
plt.plot(revenue_by_date.index, revenue_by_date.values)
plt.title('Суммарная выручка по дням')
plt.xlabel('date')
plt.ylabel('revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Для каждого товара вывести корреляцию corr(our_price, sales)
for product in PRODUCTS:
    prod_data = df[df['product'] == product]
    corr = prod_data['our_price'].corr(prod_data['sales'])
    print(f"Корреляция (our_price, sales) для '{product}': {corr:.4f}")
''')

add_md("# 2. Эвристическая модель (Rule-based)")
add_code('''
results_rules = []

for product in PRODUCTS:
    prod_data = df[df['product'] == product]
    last_row = prod_data.iloc[-1]
    avg_sales_7d = prod_data['sales'].tail(7).mean()
    current_revenue = last_row['revenue']
    
    rec_price, rule = apply_rules(last_row, avg_sales_7d)
    pred_sales, pred_revenue, delta_pct = forecast(product, rec_price, current_revenue)
    
    results_rules.append({
        'product': product,
        'our_price': last_row['our_price'],
        'rec_price': rec_price,
        'rule': rule,
        'predicted_revenue': pred_revenue,
        'delta_pct': delta_pct
    })

df_rules = pd.DataFrame(results_rules)
display(df_rules)

out_rules = data_path.parent / 'recommendations_rules.csv'
df_rules.to_csv(out_rules, index=False)
''')

add_md("# 3. Регрессионная модель (LinearRegression)")
add_code('''
results_regression = []

for product in PRODUCTS:
    prod_data = df[df['product'] == product]
    X = prod_data['our_price'].values.reshape(-1, 1)
    y = prod_data['sales'].values
    
    model = LinearRegression().fit(X, y)
    a = model.intercept_
    b = abs(model.coef_[0])
    optimal_price = round(a / (2 * b), 2)
    
    last_row = prod_data.iloc[-1]
    current_price = last_row['our_price']
    current_revenue = last_row['revenue']
    
    pred_sales, pred_revenue, delta_pct = forecast(product, optimal_price, current_revenue)
    
    print(f"[{product}] a = {a:.2f}, b = {b:.2f}, optimal_price = {optimal_price:.2f}")
    
    results_regression.append({
        'product': product,
        'current_price': current_price,
        'optimal_price': optimal_price,
        'predicted_revenue': pred_revenue,
        'delta_pct': delta_pct
    })
    
    # График: revenue(p)
    p_range = np.linspace(max(1, optimal_price * 0.5), optimal_price * 1.5, 100)
    rev_curve = p_range * (a - b * p_range)
    
    plt.figure(figsize=(10, 5))
    plt.plot(p_range, rev_curve, label='Кривая выручки')
    plt.axvline(x=current_price, color='red', linestyle='--', label=f'Текущая цена ({current_price})')
    plt.axvline(x=optimal_price, color='green', linestyle='--', label=f'Оптимальная цена ({optimal_price})')
    plt.title(f'Выручка от цены - {product}')
    plt.xlabel('Цена')
    plt.ylabel('Ожидаемая выручка')
    plt.legend()
    plt.grid(True)
    plt.show()

df_regression = pd.DataFrame(results_regression)
display(df_regression)

out_regression = data_path.parent / 'recommendations_regression.csv'
df_regression.to_csv(out_regression, index=False)
''')

add_md("# 4. Сравнение методов")
add_code('''
# Объединяем результаты
df_comparison = df_rules[['product', 'our_price']].rename(columns={'our_price': 'current_price'}).copy()
df_comparison['price_rules'] = df_rules['rec_price']
df_comparison['price_regression'] = df_regression['optimal_price']
df_comparison['revenue_rules_delta%'] = df_rules['delta_pct']
df_comparison['revenue_regression_delta%'] = df_regression['delta_pct']
display(df_comparison)

# График bar chart
plt.figure(figsize=(10, 5))
x = np.arange(len(df_comparison['product']))
width = 0.35

plt.bar(x - width/2, df_comparison['revenue_rules_delta%'], width, label='Эвристика')
plt.bar(x + width/2, df_comparison['revenue_regression_delta%'], width, label='Регрессия')

plt.xlabel('Товары')
plt.ylabel('delta_pct (%)')
plt.title('Сравнение методов: изменение выручки')
plt.xticks(x, df_comparison['product'])
plt.legend()
plt.grid(axis='y')
plt.show()

''')

add_md('''Вывод: Регрессионный метод почти во всех случаях (или везде) показывает больший или сравнимый прирост выручки, находя глобальный оптимальный уровень цены. Эвристика работает больше как локальная корректировка текущей цены на основе конкретных триггеров (изменение конкурента или просадка продаж), но не всегда максимизирует выручку.''')

add_md("# 5. Симуляция (прокрутка времени вперёд)")
add_code('''
np.random.seed(42)

sim_rules = simulate(df, n_steps=10, method='rules')
sim_regression = simulate(df, n_steps=10, method='regression')

# Оставляем только сгенерированные дни для вывода таблиц
sim_rules_generated = sim_rules.tail(50).copy() # 5 товаров * 10 шагов
sim_regression_generated = sim_regression.tail(50).copy()

print("Симуляция: Эвристика (последние 10 строк)")
display(sim_rules_generated.tail(10))

print("\\nСимуляция: Регрессия (последние 10 строк)")
display(sim_regression_generated.tail(10))

# График
plt.figure(figsize=(10, 5))
rev_rules = sim_rules.groupby('date')['revenue'].sum()
rev_regression = sim_regression.groupby('date')['revenue'].sum()
rev_rules.index = pd.to_datetime(rev_rules.index)
rev_regression.index = pd.to_datetime(rev_regression.index)

plt.plot(rev_rules.index, rev_rules.values, label='Эвристика (Симуляция)', linestyle='--')
plt.plot(rev_regression.index, rev_regression.values, label='Регрессия (Симуляция)', linestyle=':')

# Отметим границу симуляции
last_real_date = pd.to_datetime(df['date'].max())
plt.axvline(x=last_real_date, color='black', label='Начало симуляции', alpha=0.5)

plt.title('Суммарная выручка: Регрессия vs Эвристика')
plt.xlabel('Дата')
plt.ylabel('Выручка')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
''')

add_md("# 6. Итоговый отчёт")
add_code('''
best_deltas = []

for idx, row in df_comparison.iterrows():
    product = row['product']
    cur_price = row['current_price']
    
    delta_r = row['revenue_rules_delta%']
    delta_reg = row['revenue_regression_delta%']
    
    if delta_reg >= delta_r:
        best_method = 'regression'
        rec_price = row['price_regression']
        best_delta = delta_reg
    else:
        best_method = 'rules'
        rec_price = row['price_rules']
        best_delta = delta_r
        
    best_deltas.append(best_delta)
    
    print(f"{product}: текущая цена {cur_price:.2f} руб → рекомендуем {rec_price:.2f} руб")
    print(f"Метод: {best_method} | Прогноз выручки: {'+' if best_delta > 0 else ''}{best_delta:.1f}%")
    
avg_delta = np.mean(best_deltas)
print(f"\\nСуммарный прогнозируемый прирост выручки по всем товарам: {'+' if avg_delta > 0 else ''}{avg_delta:.1f}%")
''')

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebook/pricing_mvp.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"Created notebook at {out_path.absolute()}")
