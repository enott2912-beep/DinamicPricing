"""Скрипт для генерации чистого pricing_mvp.ipynb."""
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(True)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip().splitlines(True)})

# ── СЕКЦИЯ 0 ──
md("# 0. Импорты и константы")
code("""\
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
    'Хлеб':     {'base_price': 50,  'elasticity': 1.5, 'base_sales': 250},
    'Сок':      {'base_price': 120, 'elasticity': 3.0, 'base_sales': 150},
    'Кофе':     {'base_price': 450, 'elasticity': 1.2, 'base_sales': 80},
    'Шоколад':  {'base_price': 100, 'elasticity': 2.5, 'base_sales': 200},
}


def apply_rules(row, avg_sales_7d):
    \"\"\"Эвристические правила ценообразования.\"\"\"
    price = row['our_price']
    if row['competitor_price'] < price * 0.90:
        return round(price * 0.95, 2), 'competitor_undercut'
    if row['sales'] < avg_sales_7d * 0.80:
        return round(price - 1.0, 2), 'low_sales'
    return price, 'hold'


def forecast(product, rec_price, current_revenue):
    \"\"\"Прогноз продаж и выручки при рекомендованной цене.\"\"\"
    p = PRODUCTS[product]
    pred_sales = max(0, round(p['base_sales'] - p['elasticity'] * (rec_price - p['base_price'])))
    pred_revenue = pred_sales * rec_price
    delta = (pred_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0.0
    return pred_sales, round(pred_revenue, 2), round(delta, 1)


def simulate(df_src, n_steps=10, method='regression'):
    \"\"\"Симуляция на n_steps дней вперёд.\"\"\"
    np.random.seed(SEED)
    sim = df_src.copy()
    reg_prices = {}
    if method == 'regression':
        for prod in PRODUCTS:
            sub = sim[sim['product'] == prod]
            if sub.empty:
                continue
            X = sub['our_price'].values.reshape(-1, 1)
            y = sub['sales'].values
            m = LinearRegression().fit(X, y)
            a, b = m.intercept_, abs(m.coef_[0])
            reg_prices[prod] = round(a / (2 * b), 2)
    for _ in range(n_steps):
        next_date = pd.to_datetime(sim['date']).max() + pd.Timedelta(days=1)
        rows = []
        for prod, params in PRODUCTS.items():
            hist = sim[sim['product'] == prod]
            if hist.empty:
                continue
            last = hist.iloc[-1]
            if method == 'rules':
                rec, _ = apply_rules(last, hist['sales'].tail(7).mean())
            else:
                rec = reg_prices.get(prod, last['our_price'])
            dev = rec - params['base_price']
            ns = max(0, round(params['base_sales'] - params['elasticity'] * dev + np.random.normal(0, 5)))
            rows.append({'date': str(next_date.date()), 'product': prod, 'our_price': rec,
                         'competitor_price': last['competitor_price'], 'sales': ns, 'revenue': ns * rec})
        sim = pd.concat([sim, pd.DataFrame(rows)], ignore_index=True)
    return sim""")

# ── СЕКЦИЯ 1 ──
md("# 1. Загрузка и обзор данных (EDA)")
code("""\
data_path = Path('data/sales_history.csv')
if not data_path.exists():
    data_path = Path('../data/sales_history.csv')

df = pd.read_csv(data_path)
print('Shape:', df.shape)
print('\\nDtypes:\\n', df.dtypes)
display(df.head(5))
print('\\nDescribe:')
display(df.describe())

# Scatter: our_price vs sales
plt.figure(figsize=(10, 5))
for product in df['product'].unique():
    s = df[df['product'] == product]
    plt.scatter(s['our_price'], s['sales'], label=product)
plt.title('Зависимость продаж от цены')
plt.xlabel('our_price'); plt.ylabel('sales')
plt.grid(True); plt.legend(); plt.show()

# Line: суммарная выручка по дням
plt.figure(figsize=(10, 5))
rev = df.groupby('date')['revenue'].sum()
rev.index = pd.to_datetime(rev.index)
plt.plot(rev.index, rev.values)
plt.title('Суммарная выручка по дням')
plt.xlabel('date'); plt.ylabel('revenue')
plt.grid(True); plt.xticks(rotation=45); plt.show()

# Корреляции
for product in PRODUCTS:
    s = df[df['product'] == product]
    corr = s['our_price'].corr(s['sales'])
    print(f"Корреляция (our_price, sales) для '{product}': {corr:.4f}")""")

# ── СЕКЦИЯ 2 ──
md("# 2. Эвристическая модель (Rule-based)")
code("""\
results_rules = []
for product in PRODUCTS:
    sub = df[df['product'] == product]
    last = sub.iloc[-1]
    avg7 = sub['sales'].tail(7).mean()
    rec, rule = apply_rules(last, avg7)
    _, pred_rev, delta = forecast(product, rec, last['revenue'])
    results_rules.append({'product': product, 'our_price': last['our_price'],
                          'rec_price': rec, 'rule': rule,
                          'predicted_revenue': pred_rev, 'delta_pct': delta})

df_rules = pd.DataFrame(results_rules)
display(df_rules)
df_rules.to_csv(data_path.parent / 'recommendations_rules.csv', index=False)""")

# ── СЕКЦИЯ 3 ──
md("# 3. Регрессионная модель (LinearRegression)")
code("""\
results_reg = []
for product in PRODUCTS:
    sub = df[df['product'] == product]
    X = sub['our_price'].values.reshape(-1, 1)
    y = sub['sales'].values
    model = LinearRegression().fit(X, y)
    a, b = model.intercept_, abs(model.coef_[0])
    opt = round(a / (2 * b), 2)
    cur_price, cur_rev = sub.iloc[-1]['our_price'], sub.iloc[-1]['revenue']
    _, pred_rev, delta = forecast(product, opt, cur_rev)
    print(f'[{product}] a={a:.2f}, b={b:.2f}, optimal={opt:.2f}')
    results_reg.append({'product': product, 'current_price': cur_price,
                        'optimal_price': opt, 'predicted_revenue': pred_rev, 'delta_pct': delta})

    p_range = np.linspace(max(1, opt * 0.5), opt * 1.5, 100)
    plt.figure(figsize=(10, 5))
    plt.plot(p_range, p_range * (a - b * p_range), label='Кривая выручки')
    plt.axvline(cur_price, color='red', ls='--', label=f'Текущая ({cur_price})')
    plt.axvline(opt, color='green', ls='--', label=f'Оптимальная ({opt})')
    plt.title(f'Выручка от цены — {product}')
    plt.xlabel('Цена'); plt.ylabel('Выручка')
    plt.legend(); plt.grid(True); plt.show()

df_reg = pd.DataFrame(results_reg)
display(df_reg)
df_reg.to_csv(data_path.parent / 'recommendations_regression.csv', index=False)""")

# ── СЕКЦИЯ 4 ──
md("# 4. Сравнение методов")
code("""\
comp = df_rules[['product', 'our_price']].rename(columns={'our_price': 'current_price'}).copy()
comp['price_rules'] = df_rules['rec_price']
comp['price_regression'] = df_reg['optimal_price']
comp['revenue_rules_delta%'] = df_rules['delta_pct']
comp['revenue_regression_delta%'] = df_reg['delta_pct']
display(comp)

x = np.arange(len(comp)); w = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - w/2, comp['revenue_rules_delta%'], w, label='Эвристика')
plt.bar(x + w/2, comp['revenue_regression_delta%'], w, label='Регрессия')
plt.xlabel('Товары'); plt.ylabel('delta_pct (%)')
plt.title('Сравнение методов: изменение выручки')
plt.xticks(x, comp['product']); plt.legend(); plt.grid(axis='y'); plt.show()""")

md("**Вывод:** Регрессионный метод находит глобальный оптимум цены и показывает больший прирост выручки. "
   "Эвристика — локальная корректировка, полезна как оперативная реакция на рынок.")

# ── СЕКЦИЯ 5 ──
md("# 5. Симуляция (прокрутка времени вперёд)")
code("""\
np.random.seed(42)
sim_rules = simulate(df, n_steps=10, method='rules')
sim_reg = simulate(df, n_steps=10, method='regression')

print('Симуляция: Эвристика (последние 10 строк)')
display(sim_rules.tail(10))
print('\\nСимуляция: Регрессия (последние 10 строк)')
display(sim_reg.tail(10))

plt.figure(figsize=(10, 5))
for s, lbl, ls in [(sim_rules, 'Эвристика', '--'), (sim_reg, 'Регрессия', ':')]:
    r = s.groupby('date')['revenue'].sum()
    r.index = pd.to_datetime(r.index)
    plt.plot(r.index, r.values, label=lbl, ls=ls)
plt.axvline(pd.to_datetime(df['date'].max()), color='black', alpha=0.5, label='Начало симуляции')
plt.title('Суммарная выручка: история + симуляция')
plt.xlabel('Дата'); plt.ylabel('Выручка')
plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.show()""")

# ── СЕКЦИЯ 6 ──
md("# 6. Итоговый отчёт")
code("""\
best_deltas = []
for _, row in comp.iterrows():
    prod = row['product']
    dr, dreg = row['revenue_rules_delta%'], row['revenue_regression_delta%']
    if dreg >= dr:
        method, rec, delta = 'regression', row['price_regression'], dreg
    else:
        method, rec, delta = 'rules', row['price_rules'], dr
    best_deltas.append(delta)
    sign = '+' if delta > 0 else ''
    print(f"{prod}: текущая цена {row['current_price']:.2f} руб → рекомендуем {rec:.2f} руб")
    print(f"  Метод: {method} | Прогноз выручки: {sign}{delta:.1f}%")

avg = np.mean(best_deltas)
sign = '+' if avg > 0 else ''
print(f"\\nСуммарный прогнозируемый прирост выручки по всем товарам: {sign}{avg:.1f}%")""")

# ── СОХРАНЕНИЕ ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = Path("notebook/pricing_mvp.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Notebook created: {out}")
