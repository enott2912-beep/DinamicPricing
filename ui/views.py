import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model.pricing import apply_rules, fit_regression, forecast, forecast_from_regression, simulate
from ui.calendars import (
    apply_overview_date_filter,
    apply_predict_period_filter,
    get_product_df_with_period,
    render_overview_date_calendar,
    render_predict_period_calendar,
)
from ui.charts import (
    CHART_LABELS,
    CHART_LABELS_TIME,
    plot_price_vs_sales,
    plot_prices_over_time,
    render_stored_simulation,
)
from ui.data_manager import load_predict_data, save_predict_file


def render_welcome_screen() -> None:
    st.markdown(
        """
    <div class="welcome-header">
        <h3 style="margin:0;">🎯 Добро пожаловать</h3>
    </div>

    <div class="welcome-card">
        Это учебный прототип <strong>динамического ценообразования</strong>:
        вы загружаете свой файл CSV или генерируете историю продаж,
        изучаете аналитику и запускаете симуляции выручки.
    </div>

    <div class="welcome-card">
        <strong>📌 Как начать:</strong>
        <div class="welcome-step">
            <strong>Способ 1:</strong> Загрузите свой CSV с колонками:<br>
            <code class="theme-code">date, product_id, product, our_price, competitor_price, sales, revenue</code>
        </div>
        <div class="welcome-step">
            <strong>Способ 2:</strong> Настройте параметры в боковой панели и нажмите <strong>«Сгенерировать историю»</strong>.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.info("💡 Основные разделы станут доступны после загрузки или генерации данных.")


def render_overview_tab(df: pd.DataFrame, selected_product: str) -> None:
    st.title(f"📊 Анализ продаж: {selected_product}")
    prod_df_raw, _ = get_product_df_with_period(df, selected_product)
    if len(prod_df_raw) == 0:
        st.warning("Нет данных по выбранному товару.")
        st.stop()

    available_dates = set(prod_df_raw["date"].dt.date)
    with st.expander("📅 Период анализа (календарь по датам из таблицы)", expanded=True):
        render_overview_date_calendar(available_dates)
        rs = st.session_state.ov_range_start
        re = st.session_state.ov_range_end
        if re is not None:
            st.success(f"Выбран период: **{rs}** — **{re}**")
        elif st.session_state.ov_pending_second:
            st.caption(f"Показаны данные с **{rs}** до последней даты в таблице (ожидается выбор конца периода).")

    prod_df = apply_overview_date_filter(prod_df_raw)
    if len(prod_df) == 0:
        st.warning("За выбранный период нет строк. Расширьте диапазон или сбросьте период.")
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


def render_recommendations_tab(df: pd.DataFrame, selected_product: str) -> None:
    st.title(f"💡 Рекомендации по цене: {selected_product}")
    _, prod_df = get_product_df_with_period(df, selected_product)
    if len(prod_df) == 0:
        st.warning("За выбранный период нет данных по товару.")
        st.stop()

    rs = st.session_state.ov_range_start
    re = st.session_state.ov_range_end
    if re is not None:
        st.caption(f"Период расчета рекомендаций: **{rs}** — **{re}**")
    elif st.session_state.ov_pending_second:
        st.caption(f"Период расчета рекомендаций: с **{rs}** до последней даты в таблице.")

    last_row = prod_df.iloc[-1]
    prev_sales = prod_df["sales"].iloc[:-1]
    avg7 = prev_sales.tail(7).mean() if len(prev_sales) > 0 else last_row["sales"]

    rec_price_rules, rule_name = apply_rules(last_row, avg7)
    fc_rules = forecast(selected_product, rec_price_rules, last_row["revenue"])

    a, b, opt_price_reg, is_reg_reliable = fit_regression(prod_df, selected_product)
    fc_reg = forecast_from_regression(a, b, opt_price_reg, last_row["revenue"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Текущее состояние")
        st.metric("Наша цена", f"{last_row['our_price']:.2f} ₽")
        st.metric("Цена конкурента", f"{last_row['competitor_price']:.2f} ₽")
        st.metric("Выручка (день)", f"{last_row['revenue']:.2f} ₽")
    with c2:
        st.subheader("Эвристика")
        st.metric("Новая цена", f"{rec_price_rules:.2f} ₽", f"{fc_rules['growth_pct']:+.1f}% выручки")
        st.caption(f"Правило: {rule_name}")
    with c3:
        st.subheader("Регрессия")
        st.metric("Новая цена", f"{opt_price_reg:.2f} ₽", f"{fc_reg['growth_pct']:+.1f}% выручки")
        st.caption(f"Формула: Revenue = P * ({a:.1f} - {b:.2f}*P)")
        if not is_reg_reliable:
            st.warning(
                "Наклон регрессии не отрицательный: оценка эластичности ненадежна, "
                "поэтому оптимальная цена может быть неточной."
            )

    st.divider()
    st.subheader("Анализ эластичности выручки (Regression)")
    p_min = max(1, opt_price_reg * 0.5)
    p_max = opt_price_reg * 1.5
    prices = np.linspace(p_min, p_max, 100)
    revenues = prices * (a - b * prices)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices, revenues, label="Прогноз выручки", color="gray", alpha=0.5)
    ax.axvline(last_row["our_price"], color="red", ls="--", label=f"Текущая ({last_row['our_price']:.2f})")
    ax.axvline(opt_price_reg, color="green", ls="-", label=f"Оптимальная ({opt_price_reg:.2f})")
    ax.set_xlabel("Цена (₽)")
    ax.set_ylabel("Прогнозируемая выручка")
    ax.legend()
    st.pyplot(fig)

    st.info(
        f"👉 **Итог по {selected_product}**: старая цена {last_row['our_price']:.2f} ₽ → "
        f"новая цена {opt_price_reg:.2f} ₽ | прогноз выручки: {fc_reg['growth_pct']:+.1f}%"
    )


def render_simulation_tab(df: pd.DataFrame) -> None:
    st.title("🔮 Симуляция будущего")
    rs = st.session_state.get("ov_range_start")
    re = st.session_state.get("ov_range_end")
    if rs:
        period_text = f"**{rs}** — **{re if re else '...'}**"
        st.info(
            f"💡 Модели будут обучаться на историческом периоде: {period_text}. "
            "Вы можете изменить его в календаре на вкладке '📊 Обзор'."
        )

    col1, col2, col3 = st.columns(3)
    sim_scope_options = ["Все товары"] + sorted(list(df["product"].unique()))
    sim_scope = col1.selectbox("Область симуляции", sim_scope_options)
    n_steps = col2.slider("Горизонт симуляции (дней)", 7, 30, 14)
    method = col3.selectbox("Метод принятия решений", ["regression", "rules"])

    if st.button("Запустить симуляцию", type="primary"):
        _, prod_df_period = get_product_df_with_period(df, sim_scope)
        if len(prod_df_period) < 2:
            st.error("⛔ Слишком короткий период для обучения. Выберите диапазон пошире в календаре.")
            st.stop()

        with st.spinner("Рынок просчитывается..."):
            simulated_df = simulate(prod_df_period, n_steps, method, target_product=sim_scope)

        if sim_scope == "Все товары":
            hist_rev = prod_df_period.groupby("date")["revenue"].sum()
            sim_rev = simulated_df.groupby("date")["revenue"].sum()
            title_suffix = "по всем товарам"
        else:
            hist_rev = prod_df_period[prod_df_period["product"] == sim_scope].groupby("date")["revenue"].sum()
            sim_rev = simulated_df[simulated_df["product"] == sim_scope].groupby("date")["revenue"].sum()
            title_suffix = f"по товару '{sim_scope}'"

        max_period_date = prod_df_period["date"].max()
        future_sim = sim_rev[sim_rev.index > max_period_date]
        predict_rows = simulated_df[simulated_df["date"] > max_period_date].copy()
        save_predict_file(predict_rows)

        avg_hist = float(hist_rev.mean()) if len(hist_rev) else float("nan")
        avg_sim = float(future_sim.mean()) if not future_sim.empty else float("nan")
        delta = (
            (avg_sim - avg_hist) / avg_hist * 100
            if avg_hist and not np.isnan(avg_sim) and avg_hist != 0
            else 0.0
        )
        first_day_rev = float(future_sim.iloc[0]) if not future_sim.empty else float("nan")
        last_day_rev = float(future_sim.iloc[-1]) if not future_sim.empty else float("nan")

        st.session_state["sim_last"] = {
            "sim_scope": sim_scope,
            "n_steps": n_steps,
            "method": method,
            "hist_rev": hist_rev,
            "future_sim": future_sim,
            "max_period_date": max_period_date,
            "title_suffix": title_suffix,
            "avg_hist": avg_hist,
            "avg_sim": avg_sim,
            "delta": delta,
            "first_day_rev": first_day_rev,
            "last_day_rev": last_day_rev,
            "future_empty": future_sim.empty,
        }
        st.session_state["sim_show_success"] = True
        st.rerun()

    sl = st.session_state.get("sim_last")
    params_match = (
        sl is not None
        and sl["sim_scope"] == sim_scope
        and sl["n_steps"] == n_steps
        and sl["method"] == method
    )
    if sl and not params_match:
        st.caption(
            "Параметры симуляции изменились относительно последнего запуска — "
            "нажмите «Запустить симуляцию», чтобы обновить график и таблицу."
        )
    if not params_match:
        return

    if st.session_state.pop("sim_show_success", False):
        st.success("✅ Симуляция завершена!")
    render_stored_simulation(sl)

    st.subheader("📋 Детализация прогноза (из predict_sales.csv)")
    pred_df = load_predict_data()
    if pred_df is None or pred_df.empty:
        st.info("Файл predict_sales.csv пока пуст. Запустите симуляцию, чтобы увидеть прогнозные строки.")
        return

    if sim_scope != "Все товары":
        pred_df = pred_df[pred_df["product"] == sim_scope].copy()
    if pred_df.empty:
        st.info("В predict_sales.csv нет строк для выбранной области симуляции.")
        return

    scope_key = f"{sim_scope}|{sl['n_steps']}|{sl['method']}"
    with st.expander("📅 Период отображения прогноза (календарь)", expanded=True):
        render_predict_period_calendar(set(pred_df["date"].dt.date), scope_key=scope_key)
        prs = st.session_state.get("pd_range_start")
        pre = st.session_state.get("pd_range_end")
        if pre is not None:
            st.success(f"Выбран период таблицы: **{prs}** — **{pre}**")
        elif st.session_state.get("pd_pending_second"):
            st.caption(
                f"Показаны строки с **{prs}** до последней даты прогноза "
                "(ожидается второй клик для конца периода)."
            )

    pred_filtered = apply_predict_period_filter(pred_df)
    st.dataframe(pred_filtered, use_container_width=True)
