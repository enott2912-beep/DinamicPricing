import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Добавляем корень проекта в sys.path для импортов
sys.path.append(str(Path(__file__).parent.parent))
from model.pricing import PRODUCTS, SEED, apply_rules, fit_regression, forecast, simulate

# ==========================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="Dynamic Pricing MVP",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
    /* Стили для карточек метрик (совместимо с темной темой) */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = Path(__file__).parent.parent / "data" / "sales_history.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
    
    df['date'] = pd.to_datetime(df['date'])
    return df

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Управление")

# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Загрузить свой CSV (sales_history)", type=["csv"])

df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Данные не найдены. Похоже, `data/sales_history.csv` отсутствует и файл не загружен.")
    if st.button("Сгенерировать демо-данные"):
        from generator.generate_data import main as gen_main
        gen_main()
        st.rerun()
    st.stop()

st.sidebar.divider()

# Выбор товара и метода
product_list = list(df['product'].unique())
selected_product = st.sidebar.selectbox("Выберите товар", product_list)
nav = st.sidebar.radio("Раздел", ["📊 Обзор", "💡 Рекомендации", "🔮 Симуляция"])

# ==========================================
# ОСНОВНОЙ КОНТЕНТ
# ==========================================

if nav == "📊 Обзор":
    st.title(f"📊 Анализ продаж: {selected_product}")
    
    prod_df = df[df['product'] == selected_product].sort_values('date')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Средняя цена", f"{prod_df['our_price'].mean():.2f} ₽")
    col2.metric("Общие продажи", f"{prod_df['sales'].sum():,.0f} шт")
    col3.metric("Выручка", f"{prod_df['revenue'].sum():,.0f} ₽")
    
    st.subheader("Зависимость Продаж от Нашей цены")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(prod_df['our_price'], prod_df['sales'], alpha=0.6, c='#1f77b4')
    ax.set_xlabel("Цена (₽)")
    ax.set_ylabel("Продажи (шт)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.subheader("Динамика нашей цены и цены конкурента")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(prod_df['date'], prod_df['our_price'], label='Наша цена', marker='.', alpha=0.8)
    ax2.plot(prod_df['date'], prod_df['competitor_price'], label='Цена конкурента', ls='--', alpha=0.6)
    ax2.set_ylabel("Цена (₽)")
    ax2.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)

elif nav == "💡 Рекомендации":
    st.title(f"💡 Рекомендации по цене: {selected_product}")
    
    prod_df = df[df['product'] == selected_product]
    last_row = prod_df.iloc[-1]
    avg7 = prod_df['sales'].tail(7).mean()
    
    # 1. Эвристика
    rec_price_rules, rule_name = apply_rules(last_row, avg7)
    fc_rules = forecast(selected_product, rec_price_rules, last_row['revenue'])
    
    # 2. Регрессия
    a, b, opt_price_reg = fit_regression(df, selected_product)
    fc_reg = forecast(selected_product, opt_price_reg, last_row['revenue'])
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Текущее состояние")
        st.write(f"Цена: **{last_row['our_price']:.2f} ₽**")
        st.write(f"Конкурент: **{last_row['competitor_price']:.2f} ₽**")
        st.write(f"Выручка (посл. день): **{last_row['revenue']:.2f} ₽**")
        
    with c2:
        st.subheader("Эвристика")
        st.metric("Новая цена", f"{rec_price_rules:.2f} ₽", f"{fc_rules['growth_pct']:+.1f}% выручки")
        st.caption(f"Правило: {rule_name}")
        
    with c3:
        st.subheader("Регрессия")
        st.metric("Новая цена", f"{opt_price_reg:.2f} ₽", f"{fc_reg['growth_pct']:+.1f}% выручки")
        st.caption(f"Формула: Revenue = P * ({a:.1f} - {b:.2f}*P)")

    # Визуализация функции выручки
    st.divider()
    st.subheader("Анализ эластичности выручки (Regression)")
    
    p_min = max(1, opt_price_reg * 0.5)
    p_max = opt_price_reg * 1.5
    prices = np.linspace(p_min, p_max, 100)
    revenues = prices * (a - b * prices)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices, revenues, label='Прогноз выручки', color='gray', alpha=0.5)
    ax.axvline(last_row['our_price'], color='red', ls='--', label=f"Текущая ({last_row['our_price']:.2f})")
    ax.axvline(opt_price_reg, color='green', ls='-', label=f"Оптимальная ({opt_price_reg:.2f})")
    ax.set_xlabel("Цена (₽)")
    ax.set_ylabel("Прогнозируемая выручка")
    ax.legend()
    st.pyplot(fig)
    
    st.info(f"👉 **Итог по {selected_product}**: старая цена {last_row['our_price']:.2f} ₽ → новая цена {opt_price_reg:.2f} ₽ | прогноз выручки: {fc_reg['growth_pct']:+.1f}%")

elif nav == "🔮 Симуляция":
    st.title("🔮 Симуляция будущего")
    
    col1, col2 = st.columns(2)
    n_days = col1.slider("Горизонт симуляции (дней)", 7, 30, 14)
    method = col2.selectbox("Метод принятия решений", ["regression", "rules"])
    
    if st.button("Запустить симуляцию", type="primary"):
        with st.spinner("Рынок просчитывается..."):
            simulated_df = simulate(df, n_days, method)
            
        st.success(f"Симуляция на {n_days} дней по методу '{method}' завершена!")
        
        # Сравнение выручки
        hist_rev = df.groupby('date')['revenue'].sum()
        sim_rev = simulated_df.groupby('date')['revenue'].sum()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist_rev.index, hist_rev.values, label='История', color='blue')
        ax.plot(sim_rev.index[len(hist_rev)-1:], sim_rev.values[len(hist_rev)-1:], 
                label='Симуляция', color='green', ls='--')
        ax.axvline(df['date'].max(), color='black', alpha=0.2)
        ax.set_ylabel("Суммарная выручка по всем товарам (₽)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        avg_hist = hist_rev.mean()
        avg_sim = simulated_df[simulated_df['date'] > df['date'].max()].groupby('date')['revenue'].sum().mean()
        delta = (avg_sim - avg_hist) / avg_hist * 100
        
        st.metric("Средняя выручка в день (Sim vs Hist)", f"{avg_sim:,.0f} ₽", f"{delta:+.1f}%")
        
        st.subheader("Детали симуляции (последние записи)")
        st.dataframe(simulated_df.tail(10), use_container_width=True)
