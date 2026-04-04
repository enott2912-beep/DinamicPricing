"""
Streamlit UI — Алгоритм динамического ценообразования для ритейла.
Спринт 3: выбор товара, график текущей vs оптимальной цены.
Спринт 4: симуляция (прокрутка времени вперёд).
"""

import sys
from pathlib import Path

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from config import PRODUCTS, SEED

# ──────────────────────────── НАСТРОЙКИ СТРАНИЦЫ ────────────────────────────
st.set_page_config(
    page_title="Динамическое ценообразование — MVP",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────── ФУНКЦИИ ───────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """Загружает sales_history.csv из папки data/."""
    data_path = Path(__file__).parent.parent / "data" / "sales_history.csv"
    if not data_path.exists():
        st.error(f"Файл не найден: {data_path}. Сначала запустите generator/generate_data.py")
        st.stop()
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def apply_rules(row: pd.Series, avg_sales_7d: float) -> tuple[float, str]:
    """Эвристические правила для рекомендации цены."""
    our_price = row["our_price"]
    competitor_price = row["competitor_price"]
    sales = row["sales"]

    if competitor_price < our_price * 0.90:
        return round(our_price * 0.95, 2), "competitor_undercut"
    elif sales < avg_sales_7d * 0.80:
        return round(our_price - 1.0, 2), "low_sales"
    else:
        return our_price, "hold"


def fit_regression(prod_data: pd.DataFrame) -> dict:
    """Обучает LinearRegression и возвращает коэффициенты + оптимальную цену."""
    X = prod_data["our_price"].values.reshape(-1, 1)
    y = prod_data["sales"].values
    model = LinearRegression().fit(X, y)
    a = model.intercept_
    b = abs(model.coef_[0])
    optimal_price = round(a / (2 * b), 2)
    return {"a": a, "b": b, "optimal_price": optimal_price, "model": model}


def forecast(product: str, rec_price: float, current_revenue: float) -> dict:
    """Прогноз продаж и выручки при рекомендованной цене."""
    params = PRODUCTS[product]
    price_dev = rec_price - params["base_price"]
    predicted_sales = max(0, round(params["base_sales"] - params["elasticity"] * price_dev))
    predicted_revenue = predicted_sales * rec_price
    delta_pct = 0.0
    if current_revenue > 0:
        delta_pct = (predicted_revenue - current_revenue) / current_revenue * 100
    return {
        "predicted_sales": predicted_sales,
        "predicted_revenue": round(predicted_revenue, 2),
        "delta_pct": round(delta_pct, 1),
    }


def simulate(df_src: pd.DataFrame, n_steps: int, method: str) -> pd.DataFrame:
    """Симуляция на n_steps дней вперёд выбранным методом."""
    np.random.seed(SEED)
    sim_df = df_src.copy()

    reg_results = {}
    if method == "regression":
        for prod in PRODUCTS:
            sub = sim_df[sim_df["product"] == prod]
            if not sub.empty:
                reg = fit_regression(sub)
                reg_results[prod] = reg["optimal_price"]

    for _ in range(n_steps):
        last_date = sim_df["date"].max()
        next_date = last_date + pd.Timedelta(days=1)
        new_rows = []
        for prod in PRODUCTS:
            prod_hist = sim_df[sim_df["product"] == prod]
            if prod_hist.empty:
                continue
            last_row = prod_hist.iloc[-1]

            if method == "rules":
                avg_7 = prod_hist["sales"].tail(7).mean()
                rec_price, _ = apply_rules(last_row, avg_7)
            else:
                rec_price = reg_results.get(prod, last_row["our_price"])

            params = PRODUCTS[prod]
            price_dev = rec_price - params["base_price"]
            new_sales = params["base_sales"] - params["elasticity"] * price_dev + np.random.normal(0, 5)
            new_sales = max(0, round(new_sales))

            new_rows.append({
                "date": next_date,
                "product": prod,
                "our_price": rec_price,
                "competitor_price": last_row["competitor_price"],
                "sales": new_sales,
                "revenue": new_sales * rec_price,
            })
        sim_df = pd.concat([sim_df, pd.DataFrame(new_rows)], ignore_index=True)

    return sim_df


# ──────────────────────────── ЗАГРУЗКА ДАННЫХ ───────────────────────────────
df = load_data()

# ──────────────────────────── БОКОВОЕ МЕНЮ ──────────────────────────────────
st.sidebar.title("⚙️ Параметры")
page = st.sidebar.radio("Раздел", [
    "📈 Обзор данных",
    "🏷️ Рекомендации по ценам",
    "🔮 Симуляция",
])

# ──────────────────────────── СТРАНИЦА 1: ОБЗОР ─────────────────────────────
if page == "📈 Обзор данных":
    st.title("📈 Обзор исторических данных")

    product_sel = st.selectbox("Выберите товар", ["Все"] + list(PRODUCTS.keys()))

    if product_sel == "Все":
        subset = df
    else:
        subset = df[df["product"] == product_sel]

    col1, col2, col3 = st.columns(3)
    col1.metric("Записей", f"{len(subset):,}")
    col2.metric("Средняя цена", f"{subset['our_price'].mean():.2f} ₽")
    col3.metric("Средняя выручка/день", f"{subset.groupby('date')['revenue'].sum().mean():,.0f} ₽")

    # Scatter: цена vs продажи
    st.subheader("Зависимость продаж от цены")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for prod in subset["product"].unique():
        p = subset[subset["product"] == prod]
        ax1.scatter(p["our_price"], p["sales"], label=prod, alpha=0.7)
    ax1.set_xlabel("Наша цена, ₽")
    ax1.set_ylabel("Продажи, шт.")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Line: выручка по дням
    st.subheader("Суммарная выручка по дням")
    rev = subset.groupby("date")["revenue"].sum()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(rev.index, rev.values, linewidth=1.5)
    ax2.set_xlabel("Дата")
    ax2.set_ylabel("Выручка, ₽")
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Корреляции
    st.subheader("Корреляция: цена ↔ продажи")
    corr_data = []
    for prod in PRODUCTS:
        p = df[df["product"] == prod]
        corr_data.append({"Товар": prod, "Корреляция": round(p["our_price"].corr(p["sales"]), 4)})
    st.dataframe(pd.DataFrame(corr_data), use_container_width=True, hide_index=True)

# ──────────────────────────── СТРАНИЦА 2: РЕКОМЕНДАЦИИ ──────────────────────
elif page == "🏷️ Рекомендации по ценам":
    st.title("🏷️ Текущая цена vs Оптимальная цена")

    product_sel = st.selectbox("Выберите товар", list(PRODUCTS.keys()))
    prod_data = df[df["product"] == product_sel]
    last_row = prod_data.iloc[-1]
    current_price = last_row["our_price"]
    current_revenue = last_row["revenue"]

    # Эвристика
    avg_7 = prod_data["sales"].tail(7).mean()
    rule_price, rule_name = apply_rules(last_row, avg_7)
    rule_forecast = forecast(product_sel, rule_price, current_revenue)

    # Регрессия
    reg = fit_regression(prod_data)
    reg_forecast = forecast(product_sel, reg["optimal_price"], current_revenue)

    # Метрики
    col1, col2, col3 = st.columns(3)
    col1.metric("Текущая цена", f"{current_price:.2f} ₽")
    col2.metric("Эвристика", f"{rule_price:.2f} ₽", f"{rule_forecast['delta_pct']:+.1f}%")
    col3.metric("Регрессия", f"{reg['optimal_price']:.2f} ₽", f"{reg_forecast['delta_pct']:+.1f}%")

    st.caption(f"Правило эвристики: **{rule_name}**")

    # График кривой выручки
    st.subheader(f"Кривая выручки — {product_sel}")
    a, b = reg["a"], reg["b"]
    p_range = np.linspace(max(1, reg["optimal_price"] * 0.5), reg["optimal_price"] * 1.5, 200)
    rev_curve = p_range * (a - b * p_range)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(p_range, rev_curve, linewidth=2, label="Кривая выручки")
    ax.axvline(x=current_price, color="red", linestyle="--", linewidth=1.5,
               label=f"Текущая ({current_price:.2f} ₽)")
    ax.axvline(x=reg["optimal_price"], color="green", linestyle="--", linewidth=1.5,
               label=f"Оптимальная ({reg['optimal_price']:.2f} ₽)")
    ax.axvline(x=rule_price, color="blue", linestyle=":", linewidth=1.5,
               label=f"Эвристика ({rule_price:.2f} ₽)")
    ax.set_xlabel("Цена, ₽")
    ax.set_ylabel("Ожидаемая выручка, ₽")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Сводная таблица по всем товарам
    st.subheader("Сводка по всем товарам")
    rows = []
    for prod in PRODUCTS:
        pdata = df[df["product"] == prod]
        lr = pdata.iloc[-1]
        avg7 = pdata["sales"].tail(7).mean()
        rp, rn = apply_rules(lr, avg7)
        rf = forecast(prod, rp, lr["revenue"])
        rg = fit_regression(pdata)
        rgf = forecast(prod, rg["optimal_price"], lr["revenue"])
        rows.append({
            "Товар": prod,
            "Текущая цена": f"{lr['our_price']:.2f}",
            "Эвристика": f"{rp:.2f}",
            "Регрессия": f"{rg['optimal_price']:.2f}",
            "Δ выручки (эвр.)": f"{rf['delta_pct']:+.1f}%",
            "Δ выручки (рег.)": f"{rgf['delta_pct']:+.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ──────────────────────────── СТРАНИЦА 3: СИМУЛЯЦИЯ ─────────────────────────
elif page == "🔮 Симуляция":
    st.title("🔮 Симуляция: прокрутка времени вперёд")

    col1, col2 = st.columns(2)
    with col1:
        n_steps = st.slider("Количество дней симуляции", 1, 30, 10)
    with col2:
        method = st.radio("Метод ценообразования", ["regression", "rules"], horizontal=True)

    if st.button("▶️ Запустить симуляцию", type="primary"):
        with st.spinner("Симуляция выполняется..."):
            sim_df = simulate(df, n_steps=n_steps, method=method)

        # Новые строки симуляции
        original_max_date = df["date"].max()
        sim_only = sim_df[sim_df["date"] > original_max_date]

        st.success(f"Симуляция завершена: +{n_steps} дней, метод «{method}»")

        # Таблица результатов
        st.subheader("Результаты симуляции")
        st.dataframe(sim_only.round(2), use_container_width=True, hide_index=True)

        # График выручки
        st.subheader("Суммарная выручка: история + симуляция")
        rev_all = sim_df.groupby("date")["revenue"].sum()
        rev_hist = df.groupby("date")["revenue"].sum()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rev_hist.index, rev_hist.values, label="История", linewidth=1.5)
        ax.plot(rev_all.index, rev_all.values, label="С симуляцией", linewidth=1.5, linestyle="--")
        ax.axvline(x=original_max_date, color="black", alpha=0.4, linestyle=":", label="Начало симуляции")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Выручка, ₽")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Итоговый прогноз
        st.subheader("Прогноз выручки")
        avg_hist = rev_hist.mean()
        avg_sim = sim_only.groupby("date")["revenue"].sum().mean()
        delta = (avg_sim - avg_hist) / avg_hist * 100

        st.metric(
            "Средняя дневная выручка (симуляция vs история)",
            f"{avg_sim:,.0f} ₽",
            f"{delta:+.1f}%",
        )
