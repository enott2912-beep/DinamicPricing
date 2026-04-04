"""
Streamlit UI — Динамическое ценообразование для ритейла.

Три раздела:
  1. Обзор данных (EDA)
  2. Рекомендации по ценам (эвристика + регрессия)
  3. Симуляция (прокрутка времени вперёд)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from config import PRODUCTS, SEED


# ─────────────────────── Настройки страницы ─────────────────────────────────

st.set_page_config(
    page_title="Динамическое ценообразование — MVP",
    page_icon="📊",
    layout="wide",
)


# ─────────────────────── Функции ────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """Загружает sales_history.csv."""
    path = Path(__file__).parent.parent / "data" / "sales_history.csv"
    if not path.exists():
        st.error(f"Файл не найден: {path}. Запустите generator/generate_data.py")
        st.stop()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def apply_rules(row: pd.Series, avg_sales_7d: float) -> tuple[float, str]:
    """Эвристические правила ценообразования."""
    price = row["our_price"]
    if row["competitor_price"] < price * 0.90:
        return round(price * 0.95, 2), "competitor_undercut"
    if row["sales"] < avg_sales_7d * 0.80:
        return round(price - 1.0, 2), "low_sales"
    return price, "hold"


def fit_regression(prod_data: pd.DataFrame) -> dict:
    """Обучает LinearRegression, возвращает коэффициенты и оптимальную цену."""
    X = prod_data["our_price"].values.reshape(-1, 1)
    y = prod_data["sales"].values
    model = LinearRegression().fit(X, y)
    a = model.intercept_
    b = abs(model.coef_[0])
    return {"a": a, "b": b, "optimal_price": round(a / (2 * b), 2)}


def forecast(product: str, rec_price: float, current_revenue: float) -> dict:
    """Прогноз продаж и выручки при рекомендованной цене."""
    p = PRODUCTS[product]
    pred_sales = max(0, round(p["base_sales"] - p["elasticity"] * (rec_price - p["base_price"])))
    pred_revenue = pred_sales * rec_price
    delta = (pred_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0.0
    return {"predicted_sales": pred_sales, "predicted_revenue": round(pred_revenue, 2), "delta_pct": round(delta, 1)}


def simulate(df_src: pd.DataFrame, n_steps: int, method: str) -> pd.DataFrame:
    """Симуляция на n_steps дней вперёд."""
    np.random.seed(SEED)
    sim = df_src.copy()

    # Предрасчёт оптимальных цен для регрессии
    reg_prices = {}
    if method == "regression":
        for prod in PRODUCTS:
            sub = sim[sim["product"] == prod]
            if not sub.empty:
                reg_prices[prod] = fit_regression(sub)["optimal_price"]

    for _ in range(n_steps):
        next_date = sim["date"].max() + pd.Timedelta(days=1)
        rows = []
        for prod, params in PRODUCTS.items():
            hist = sim[sim["product"] == prod]
            if hist.empty:
                continue
            last = hist.iloc[-1]

            if method == "rules":
                rec_price, _ = apply_rules(last, hist["sales"].tail(7).mean())
            else:
                rec_price = reg_prices.get(prod, last["our_price"])

            dev = rec_price - params["base_price"]
            new_sales = max(0, round(params["base_sales"] - params["elasticity"] * dev + np.random.normal(0, 5)))

            rows.append({
                "date": next_date, "product": prod,
                "our_price": rec_price, "competitor_price": last["competitor_price"],
                "sales": new_sales, "revenue": new_sales * rec_price,
            })
        sim = pd.concat([sim, pd.DataFrame(rows)], ignore_index=True)

    return sim


# ─────────────────────── Данные ─────────────────────────────────────────────

df = load_data()

# ─────────────────────── Боковое меню ───────────────────────────────────────

st.sidebar.title("⚙️ Параметры")
page = st.sidebar.radio("Раздел", ["📈 Обзор данных", "🏷️ Рекомендации по ценам", "🔮 Симуляция"])

# ─────────────────────── Страница 1: Обзор ─────────────────────────────────

if page == "📈 Обзор данных":
    st.title("📈 Обзор исторических данных")
    product_sel = st.selectbox("Товар", ["Все"] + list(PRODUCTS.keys()))
    subset = df if product_sel == "Все" else df[df["product"] == product_sel]

    c1, c2, c3 = st.columns(3)
    c1.metric("Записей", f"{len(subset):,}")
    c2.metric("Средняя цена", f"{subset['our_price'].mean():.2f} ₽")
    c3.metric("Средняя выручка/день", f"{subset.groupby('date')['revenue'].sum().mean():,.0f} ₽")

    st.subheader("Зависимость продаж от цены")
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in subset["product"].unique():
        s = subset[subset["product"] == p]
        ax.scatter(s["our_price"], s["sales"], label=p, alpha=0.7)
    ax.set(xlabel="Цена, ₽", ylabel="Продажи, шт.")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Суммарная выручка по дням")
    rev = subset.groupby("date")["revenue"].sum()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(rev.index, rev.values, linewidth=1.5)
    ax2.set(xlabel="Дата", ylabel="Выручка, ₽"); ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Корреляция: цена ↔ продажи")
    corrs = [{"Товар": p, "Корреляция": round(df[df["product"] == p]["our_price"].corr(df[df["product"] == p]["sales"]), 4)} for p in PRODUCTS]
    st.dataframe(pd.DataFrame(corrs), use_container_width=True, hide_index=True)

# ─────────────────────── Страница 2: Рекомендации ──────────────────────────

elif page == "🏷️ Рекомендации по ценам":
    st.title("🏷️ Текущая цена vs Оптимальная цена")
    product_sel = st.selectbox("Товар", list(PRODUCTS.keys()))
    prod_data = df[df["product"] == product_sel]
    last = prod_data.iloc[-1]

    # Эвристика
    avg7 = prod_data["sales"].tail(7).mean()
    rule_price, rule_name = apply_rules(last, avg7)
    rule_fc = forecast(product_sel, rule_price, last["revenue"])

    # Регрессия
    reg = fit_regression(prod_data)
    reg_fc = forecast(product_sel, reg["optimal_price"], last["revenue"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Текущая цена", f"{last['our_price']:.2f} ₽")
    c2.metric("Эвристика", f"{rule_price:.2f} ₽", f"{rule_fc['delta_pct']:+.1f}%")
    c3.metric("Регрессия", f"{reg['optimal_price']:.2f} ₽", f"{reg_fc['delta_pct']:+.1f}%")
    st.caption(f"Правило эвристики: **{rule_name}**")

    # Кривая выручки
    st.subheader(f"Кривая выручки — {product_sel}")
    a, b = reg["a"], reg["b"]
    p_range = np.linspace(max(1, reg["optimal_price"] * 0.5), reg["optimal_price"] * 1.5, 200)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(p_range, p_range * (a - b * p_range), linewidth=2, label="Кривая выручки")
    ax.axvline(last["our_price"], color="red", ls="--", label=f"Текущая ({last['our_price']:.2f} ₽)")
    ax.axvline(reg["optimal_price"], color="green", ls="--", label=f"Оптимальная ({reg['optimal_price']:.2f} ₽)")
    ax.axvline(rule_price, color="blue", ls=":", label=f"Эвристика ({rule_price:.2f} ₽)")
    ax.set(xlabel="Цена, ₽", ylabel="Выручка, ₽"); ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Сводка по всем
    st.subheader("Сводка по всем товарам")
    rows = []
    for prod in PRODUCTS:
        pd_ = df[df["product"] == prod]
        lr = pd_.iloc[-1]
        rp, rn = apply_rules(lr, pd_["sales"].tail(7).mean())
        rf = forecast(prod, rp, lr["revenue"])
        rg = fit_regression(pd_)
        rgf = forecast(prod, rg["optimal_price"], lr["revenue"])
        rows.append({"Товар": prod, "Текущая": f"{lr['our_price']:.2f}", "Эвристика": f"{rp:.2f}",
                      "Регрессия": f"{rg['optimal_price']:.2f}", "Δ% эвр.": f"{rf['delta_pct']:+.1f}",
                      "Δ% рег.": f"{rgf['delta_pct']:+.1f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────── Страница 3: Симуляция ──────────────────────────────

elif page == "🔮 Симуляция":
    st.title("🔮 Симуляция: прокрутка времени вперёд")

    c1, c2 = st.columns(2)
    n_steps = c1.slider("Дней симуляции", 1, 30, 10)
    method = c2.radio("Метод", ["regression", "rules"], horizontal=True)

    if st.button("▶️ Запустить", type="primary"):
        with st.spinner("Симуляция..."):
            sim = simulate(df, n_steps=n_steps, method=method)

        border = df["date"].max()
        sim_only = sim[sim["date"] > border]

        st.success(f"Готово: +{n_steps} дней, метод «{method}»")
        st.dataframe(sim_only.round(2), use_container_width=True, hide_index=True)

        st.subheader("Выручка: история + симуляция")
        rev_all = sim.groupby("date")["revenue"].sum()
        rev_hist = df.groupby("date")["revenue"].sum()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rev_hist.index, rev_hist.values, label="История", linewidth=1.5)
        ax.plot(rev_all.index, rev_all.values, ls="--", label="С симуляцией", linewidth=1.5)
        ax.axvline(border, color="black", alpha=0.4, ls=":", label="Начало симуляции")
        ax.set(xlabel="Дата", ylabel="Выручка, ₽"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        avg_hist = rev_hist.mean()
        avg_sim = sim_only.groupby("date")["revenue"].sum().mean()
        delta = (avg_sim - avg_hist) / avg_hist * 100
        st.metric("Средняя дневная выручка (симуляция vs история)", f"{avg_sim:,.0f} ₽", f"{delta:+.1f}%")
