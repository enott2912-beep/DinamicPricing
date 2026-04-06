import sys
import calendar
from datetime import date
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Добавляем корень проекта в sys.path для импортов
sys.path.append(str(Path(__file__).parent.parent))
from model.pricing import (
    PRODUCTS,
    SEED,
    apply_rules,
    fit_regression,
    forecast,
    simulate,
)

# ==========================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(page_title="Dynamic Pricing MVP", page_icon="💰", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

    df["date"] = pd.to_datetime(df["date"])
    return df


CHART_LABELS = [
    "Точечный",
    "Линейный",
    "Круговая диаграмма",
    "Столбчатая диаграмма",
    "Гистограмма",
]
CHART_LABELS_TIME = ["Точечный", "Линейный", "Столбчатая диаграмма", "Гистограмма"]


_MONTHS_RU = (
    "",
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
)


def _init_overview_session(product_key: str, available_dates: set) -> None:
    """Инициализирует состояние календаря при смене товара или первом запуске."""
    min_d, max_d = min(available_dates), max(available_dates)
    if st.session_state.get("ov_product_key") != product_key:
        st.session_state.ov_product_key = product_key
        st.session_state.ov_range_start = min_d
        st.session_state.ov_range_end = max_d
        st.session_state.ov_pending_second = False
        st.session_state.ov_cal_year = min_d.year
        st.session_state.ov_cal_month = min_d.month


def _on_overview_day_click(clicked: date) -> None:
    """Первый клик — начало периода, второй — конец (даты из таблицы)."""
    if st.session_state.get("ov_pending_second"):
        a, b = st.session_state.ov_range_start, clicked
        if b < a:
            a, b = b, a
        st.session_state.ov_range_start = a
        st.session_state.ov_range_end = b
        st.session_state.ov_pending_second = False
    else:
        st.session_state.ov_range_start = clicked
        st.session_state.ov_range_end = None
        st.session_state.ov_pending_second = True


def _nav_overview_month(delta: int) -> None:
    y, m = st.session_state.ov_cal_year, st.session_state.ov_cal_month
    m += delta
    while m < 1:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    st.session_state.ov_cal_year = y
    st.session_state.ov_cal_month = m


def render_overview_date_calendar(available_dates: set) -> None:
    """Календарь: активны только дни, присутствующие в таблице."""
    min_d, max_d = min(available_dates), max(available_dates)
    y, m = st.session_state.ov_cal_year, st.session_state.ov_cal_month
    rs = st.session_state.ov_range_start
    re = st.session_state.ov_range_end
    pending = st.session_state.ov_pending_second
    range_end = max_d if (pending and re is None) else re
    st.caption(
        "Выберите **начальную** дату кликом, затем **конечную**. "
        "Доступны только дни, для которых есть строки в данных."
    )
    if st.session_state.ov_pending_second and st.session_state.ov_range_end is None:
        st.info(
            f"Выбрано начало: **{st.session_state.ov_range_start}**. "
            "Кликните конечную дату."
        )

    nav1, nav2, nav3 = st.columns([1, 4, 1])
    with nav1:
        if st.button("◀", key="ov_cal_prev", help="Предыдущий месяц"):
            _nav_overview_month(-1)
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='text-align:center'><b>{_MONTHS_RU[m]} {y}</b></div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("▶", key="ov_cal_next", help="Следующий месяц"):
            _nav_overview_month(1)
            st.rerun()

    weekdays = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    header = st.columns(7)
    for i, w in enumerate(weekdays):
        header[i].markdown(
            f"<div style='text-align:center;font-size:0.85em'>{w}</div>",
            unsafe_allow_html=True,
        )

    first_weekday, days_in_month = calendar.monthrange(y, m)
    # monthrange: Monday=0
    grid = []
    row = [None] * first_weekday
    for d in range(1, days_in_month + 1):
        day_date = date(y, m, d)
        row.append(day_date)
        if len(row) == 7:
            grid.append(row)
            row = []
    if row:
        while len(row) < 7:
            row.append(None)
        grid.append(row)

    for row in grid:
        cols = st.columns(7)
        for col, day_date in zip(cols, row):
            if day_date is None:
                col.empty()
                continue
            in_data = day_date in available_dates
            if not in_data:
                col.markdown(
                    f"<div style='text-align:center;color:#ccc;padding:6px'>{day_date.day}</div>",
                    unsafe_allow_html=True,
                )
                continue
            label = str(day_date.day)
            is_in_range = (
                rs is not None and range_end is not None and rs <= day_date <= range_end
            )
            col.button(
                label,
                key=f"ovday_{day_date.isoformat()}",
                on_click=_on_overview_day_click,
                args=(day_date,),
                type="primary" if is_in_range else "secondary",
                use_container_width=True,
            )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Сбросить на весь период", key="ov_reset_range"):
            st.session_state.ov_range_start = min_d
            st.session_state.ov_range_end = max_d
            st.session_state.ov_pending_second = False
            st.rerun()
    with b2:
        st.caption(f"В данных: **{min_d}** — **{max_d}**")


def apply_overview_date_filter(prod_df: pd.DataFrame) -> pd.DataFrame:
    """Фильтр по выбранному периоду (пока не выбран конец — от выбранного начала до конца ряда)."""
    rs = st.session_state.ov_range_start
    re = st.session_state.ov_range_end
    pending = st.session_state.ov_pending_second
    dcol = prod_df["date"].dt.date
    if pending and re is None:
        return prod_df[dcol >= rs].copy()
    if rs is not None and re is not None:
        return prod_df[(dcol >= rs) & (dcol <= re)].copy()
    return prod_df.copy()


def plot_price_vs_sales(ax, prod_df: pd.DataFrame, kind: str) -> None:
    """График «цена — продажи» в зависимости от типа."""
    x = prod_df["our_price"].values
    y = prod_df["sales"].values
    if len(prod_df) == 0:
        ax.text(
            0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes
        )
        return

    if kind == "Точечный":
        ax.scatter(x, y, alpha=0.6, c="#1f77b4")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Продажи (шт)")
    elif kind == "Линейный":
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker=".", alpha=0.8, color="#1f77b4")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Продажи (шт)")
    elif kind == "Столбчатая диаграмма":
        g = (
            prod_df.groupby("our_price", as_index=False)["sales"]
            .sum()
            .sort_values("our_price")
        )
        ax.bar(g["our_price"].astype(str), g["sales"], color="#1f77b4", alpha=0.85)
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Продажи (шт)")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Круговая диаграмма":
        g = prod_df.groupby("our_price", as_index=False)["sales"].sum()
        labels = [f"{row.our_price} ₽" for _, row in g.iterrows()]
        ax.pie(g["sales"], labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8})
        ax.set_ylabel("")
    elif kind == "Гистограмма":
        bins = min(20, max(5, len(prod_df) // 3))
        ax.hist(x, bins=bins, weights=y, color="#1f77b4", alpha=0.75, edgecolor="white")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Сумма продаж (вес по дням)")
    if kind != "Круговая диаграмма":
        ax.grid(True, alpha=0.3)


def plot_prices_over_time(ax, prod_df: pd.DataFrame, kind: str) -> None:
    """Динамика цен по датам."""
    if len(prod_df) == 0:
        ax.text(
            0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes
        )
        return
    t = prod_df["date"].values
    p1 = prod_df["our_price"].values
    p2 = prod_df["competitor_price"].values

    if kind == "Точечный":
        ax.scatter(t, p1, label="Наша цена", alpha=0.8, c="#1f77b4")
        ax.scatter(t, p2, label="Цена конкурента", alpha=0.6, c="#ff7f0e", marker="s")
        ax.set_ylabel("Цена (₽)")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Линейный":
        ax.plot(t, p1, label="Наша цена", marker=".", alpha=0.8)
        ax.plot(t, p2, label="Цена конкурента", ls="--", alpha=0.6)
        ax.set_ylabel("Цена (₽)")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Столбчатая диаграмма":
        x = np.arange(len(prod_df))
        w = 0.35
        ax.bar(x - w / 2, p1, width=w, label="Наша цена", alpha=0.85)
        ax.bar(x + w / 2, p2, width=w, label="Конкурент", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [pd.Timestamp(ti).strftime("%m-%d") for ti in t], rotation=45, ha="right"
        )
        ax.set_ylabel("Цена (₽)")
        ax.legend()
    elif kind == "Гистограмма":
        ax.hist(
            p1,
            bins=min(15, max(5, len(prod_df) // 2)),
            alpha=0.7,
            label="Наша цена",
            color="#1f77b4",
        )
        ax.hist(
            p2,
            bins=min(15, max(5, len(prod_df) // 2)),
            alpha=0.5,
            label="Конкурент",
            color="#ff7f0e",
        )
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Число дней")
        ax.legend()
    if kind != "Круговая диаграмма":
        ax.grid(True, alpha=0.3)


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Управление")

# Загрузка файла
uploaded_file = st.sidebar.file_uploader(
    "Загрузить свой CSV (sales_history)", type=["csv"]
)

df = load_data(uploaded_file)

if df is None:
    st.warning(
        "⚠️ Данные не найдены. Похоже, `data/sales_history.csv` отсутствует и файл не загружен."
    )
    if st.button("Сгенерировать демо-данные"):
        from generator.generate_data import main as gen_main

        gen_main()
        st.rerun()
    st.stop()

st.sidebar.divider()

# Выбор товара и метода
product_list = list(df["product"].unique())
selected_product = st.sidebar.selectbox("Выберите товар", product_list)
nav = st.sidebar.radio("Раздел", ["📊 Обзор", "💡 Рекомендации", "🔮 Симуляция"])

# ==========================================
# ОСНОВНОЙ КОНТЕНТ
# ==========================================

if nav == "📊 Обзор":
    st.title(f"📊 Анализ продаж: {selected_product}")

    prod_df_raw = df[df["product"] == selected_product].sort_values("date")
    if len(prod_df_raw) == 0:
        st.warning("Нет данных по выбранному товару.")
        st.stop()
    available_dates = set(prod_df_raw["date"].dt.date)
    _init_overview_session(selected_product, available_dates)

    with st.expander(
        "📅 Период анализа (календарь по датам из таблицы)", expanded=True
    ):
        render_overview_date_calendar(available_dates)
        rs = st.session_state.ov_range_start
        re = st.session_state.ov_range_end
        if re is not None:
            st.success(f"Выбран период: **{rs}** — **{re}**")
        elif st.session_state.ov_pending_second:
            st.caption(
                f"Показаны данные с **{rs}** до последней даты в таблице (ожидается выбор конца периода)."
            )

    prod_df = apply_overview_date_filter(prod_df_raw)

    if len(prod_df) == 0:
        st.warning(
            "За выбранный период нет строк. Расширьте диапазон или сбросьте период."
        )
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Средняя цена", f"{prod_df['our_price'].mean():.2f} ₽")
    c2.metric("Общие продажи", f"{prod_df['sales'].sum():,.0f} шт")
    c3.metric("Выручка", f"{prod_df['revenue'].sum():,.0f} ₽")

    st.subheader("Зависимость продаж от нашей цены")
    kind1 = st.selectbox("Тип графика", CHART_LABELS, key="ov_chart_price_sales")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_price_vs_sales(ax, prod_df, kind1)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Динамика нашей цены и цены конкурента")
    kind2 = st.selectbox("Тип графика", CHART_LABELS_TIME, key="ov_chart_time")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_prices_over_time(ax2, prod_df, kind2)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

elif nav == "💡 Рекомендации":
    st.title(f"💡 Рекомендации по цене: {selected_product}")

    prod_df = df[df["product"] == selected_product]
    last_row = prod_df.iloc[-1]
    avg7 = prod_df["sales"].tail(7).mean()

    # 1. Эвристика
    rec_price_rules, rule_name = apply_rules(last_row, avg7)
    fc_rules = forecast(selected_product, rec_price_rules, last_row["revenue"])

    # 2. Регрессия
    a, b, opt_price_reg = fit_regression(df, selected_product)
    fc_reg = forecast(selected_product, opt_price_reg, last_row["revenue"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Текущее состояние")
        st.metric("Наша цена", f"{last_row['our_price']:.2f} ₽")
        st.metric("Цена конкурента", f"{last_row['competitor_price']:.2f} ₽")
        st.metric("Выручка (день)", f"{last_row['revenue']:.2f} ₽")

    with c2:
        st.subheader("Эвристика")
        st.metric(
            "Новая цена",
            f"{rec_price_rules:.2f} ₽",
            f"{fc_rules['growth_pct']:+.1f}% выручки",
        )
        st.caption(f"Правило: {rule_name}")

    with c3:
        st.subheader("Регрессия")
        st.metric(
            "Новая цена",
            f"{opt_price_reg:.2f} ₽",
            f"{fc_reg['growth_pct']:+.1f}% выручки",
        )
        st.caption(f"Формула: Revenue = P * ({a:.1f} - {b:.2f}*P)")

    # Визуализация функции выручки
    st.divider()
    st.subheader("Анализ эластичности выручки (Regression)")

    p_min = max(1, opt_price_reg * 0.5)
    p_max = opt_price_reg * 1.5
    prices = np.linspace(p_min, p_max, 100)
    revenues = prices * (a - b * prices)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices, revenues, label="Прогноз выручки", color="gray", alpha=0.5)
    ax.axvline(
        last_row["our_price"],
        color="red",
        ls="--",
        label=f"Текущая ({last_row['our_price']:.2f})",
    )
    ax.axvline(
        opt_price_reg, color="green", ls="-", label=f"Оптимальная ({opt_price_reg:.2f})"
    )
    ax.set_xlabel("Цена (₽)")
    ax.set_ylabel("Прогнозируемая выручка")
    ax.legend()
    st.pyplot(fig)

    st.info(
        f"👉 **Итог по {selected_product}**: старая цена {last_row['our_price']:.2f} ₽ → новая цена {opt_price_reg:.2f} ₽ | прогноз выручки: {fc_reg['growth_pct']:+.1f}%"
    )

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
        hist_rev = df.groupby("date")["revenue"].sum()
        sim_rev = simulated_df.groupby("date")["revenue"].sum()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist_rev.index, hist_rev.values, label="История", color="blue")
        ax.plot(
            sim_rev.index[len(hist_rev) - 1 :],
            sim_rev.values[len(hist_rev) - 1 :],
            label="Симуляция",
            color="green",
            ls="--",
        )
        ax.axvline(df["date"].max(), color="black", alpha=0.2)
        ax.set_ylabel("Суммарная выручка по всем товарам (₽)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        avg_hist = hist_rev.mean()
        avg_sim = (
            simulated_df[simulated_df["date"] > df["date"].max()]
            .groupby("date")["revenue"]
            .sum()
            .mean()
        )
        delta = (avg_sim - avg_hist) / avg_hist * 100

        st.metric(
            "Средняя выручка в день (Sim vs Hist)",
            f"{avg_sim:,.0f} ₽",
            f"{delta:+.1f}%",
        )

        st.subheader("Детали симуляции (последние записи)")
        st.dataframe(simulated_df.tail(10), use_container_width=True)
