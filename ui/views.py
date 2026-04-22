import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model.pricing import (
    apply_rules,
    fit_regression_aggregate_daily,
    forecast,
    products_in_dataframe,
    simulate,
    PRODUCTS,
)
from model.analytics import get_recommendations_all_products, get_recommendations_single_product
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


def _render_welcome_demo_charts() -> None:
    """Иллюстративные графики без загрузки данных (синтетика)."""
    rng = np.random.default_rng(42)
    n = 42
    t = np.linspace(0, 3.3, n)
    price = 70 + 16 * np.sin(t) + rng.normal(0, 2.0, n)
    sales = np.clip(620 - 5.5 * price + rng.normal(0, 32, n), 35, None)
    comp = price * 1.03 + rng.normal(0, 2.2, n)
    dates = pd.date_range(periods=n, freq="D", end=pd.Timestamp("2026-03-15"))

    demo_face = "#f0f4fb"
    demo_grid = "#d8e0ef"

    fig1, ax1 = plt.subplots(figsize=(5.2, 3.1))
    ax1.set_facecolor(demo_face)
    fig1.patch.set_facecolor("#fafbfd")
    ax1.scatter(price, sales, alpha=0.72, c="#2b6fcf", edgecolors="white", linewidths=0.35, s=52, zorder=3)
    order = np.argsort(price)
    ax1.plot(price[order], sales[order], color="#153a7a", alpha=0.42, lw=1.35, zorder=2)
    ax1.set_xlabel("Наша цена (₽)", fontsize=9)
    ax1.set_ylabel("Продажи (шт)", fontsize=9)
    ax1.set_title("Обзор: цена и спрос", fontsize=10, fontweight="600", pad=10, color="#1a2744")
    ax1.grid(True, alpha=0.35, color=demo_grid)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5.2, 3.1))
    ax2.set_facecolor(demo_face)
    fig2.patch.set_facecolor("#fafbfd")
    ax2.plot(dates, price, marker=".", markersize=5, label="Наша цена", color="#2b6fcf", lw=1.2)
    ax2.plot(dates, comp, ls="--", lw=1.1, label="Конкурент", color="#c45c12", alpha=0.88)
    ax2.set_ylabel("Цена (₽)", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax2.set_title("Динамика относительно рынка", fontsize=10, fontweight="600", pad=10, color="#1a2744")
    ax2.grid(True, alpha=0.35, color=demo_grid)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.autofmt_xdate()
    fig2.tight_layout()

    c_a, c_b = st.columns(2, gap="medium")
    with c_a:
        st.markdown(
            """
            <div class="welcome-demo-wrap">
                <p class="welcome-demo-title">Пример: «цена → продажи»</p>
                <p class="welcome-demo-desc">После загрузки данных здесь будут ваши точки; для «всех товаров»
                графики можно сводить по дням.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.pyplot(fig1)
    with c_b:
        st.markdown(
            """
            <div class="welcome-demo-wrap">
                <p class="welcome-demo-title">Пример: цены во времени</p>
                <p class="welcome-demo-desc">В приложении доступен календарь периода и отдельный блок симуляции
                прогноза выручки.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.pyplot(fig2)

    plt.close(fig1)
    plt.close(fig2)


def render_welcome_screen() -> None:
    st.markdown(
        """
        <div class="welcome-hero">
            <h1>Динамическое ценообразование</h1>
            <p class="welcome-hero-sub">
                Учебный прототип: загрузите историю продаж или сгенерируйте её одной кнопкой,
                выберите товар или портфель целиком, затем изучайте метрики, рекомендации по цене и сценарный прогноз.
            </p>
            <div class="welcome-hero-badges">
                <span class="welcome-pill">📊 Обзор и календарь периода</span>
                <span class="welcome-pill">💡 Правила и регрессия</span>
                <span class="welcome-pill">🔮 Симуляция выручки</span>
                <span class="welcome-pill">📁 Свой CSV или генерация</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="welcome-section-title">Возможности приложения</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">📊</div>
                <h3>Обзор продаж</h3>
                <p>Метрики по периоду, графики «цена — спрос» и динамика относительно конкурента.
                Удобный календарь, чтобы сузить интервал анализа.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">💡</div>
                <h3>Рекомендации</h3>
                <p>Эвристики по правилам и оценка оптимальной цены через регрессию спроса;
                сравнение сценариев роста выручки.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">🔮</div>
                <h3>Симуляция</h3>
                <p>Пошаговый прогноз рынка на выбранный горизонт: по одному SKU или сразу по всем товарам,
                с сохранением прогноза в таблицу.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<p class="welcome-section-title">Пошаговый сценарий</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="welcome-timeline-wrap">
            <div class="welcome-timeline">
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">1</span>
                    <div class="welcome-timeline-body">
                        <strong>Данные</strong>
                        <span>В боковой панели загрузите CSV с историей или нажмите «Сгенерировать историю»
                        — файл продаж будет готов к анализу.</span>
                    </div>
                </div>
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">2</span>
                    <div class="welcome-timeline-body">
                        <strong>Товар</strong>
                        <span>Выберите конкретный SKU или пункт «Все товары», чтобы смотреть агрегаты
                        портфеля и общие рекомендации.</span>
                    </div>
                </div>
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">3</span>
                    <div class="welcome-timeline-body">
                        <strong>Обзор</strong>
                        <span>Откройте вкладку «Обзор»: при необходимости уточните период в календаре,
                        изучите графики и ключевые показатели.</span>
                    </div>
                </div>
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">4</span>
                    <div class="welcome-timeline-body">
                        <strong>Рекомендации</strong>
                        <span>На вкладке «Рекомендации» посмотрите предложенную цену по правилам и по модели,
                        подсказку по эластичности.</span>
                    </div>
                </div>
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">5</span>
                    <div class="welcome-timeline-body">
                        <strong>Симуляция</strong>
                        <span>Запустите симуляцию на вкладке «Симуляция», сравните историческую и прогнозную
                        выручку, при необходимости выгрузите детализацию из сохранённого прогноза.</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="welcome-section-title">Пример визуализаций</p>', unsafe_allow_html=True)
    _render_welcome_demo_charts()

    with st.expander("📎 Формат CSV и быстрый старт", expanded=False):
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-step">
                    <strong>Способ 1 — свой файл:</strong> колонки
                    <code class="theme-code">date, store_id, store, store_profile, brand_id, brand, product_id, product, our_price, competitor_1_price, competitor_2_price, competitor_price, is_oos, sales, revenue, cogs, profit</code>
                </div>
                <div class="welcome-step">
                    <strong>Способ 2 — без файла:</strong> в боковой панели задайте число дней и нажмите
                    <strong>«Сгенерировать историю»</strong> — будет создан учебный набор продаж.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="welcome-cta">
            <strong>Готово начать?</strong><br>
            Загрузите данные слева или сгенерируйте историю — разделы «Обзор», «Рекомендации» и «Симуляция»
            откроются автоматически.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab(df: pd.DataFrame, selected_product: str) -> None:
    st.title(f"📊 Анализ продаж: {selected_product}")
    prod_df_raw, _ = get_product_df_with_period(df, selected_product)
    if len(prod_df_raw) == 0:
        st.warning("Нет данных по выбранному товару.")
        st.stop()
    if selected_product == "Все товары":
        st.caption(
            "Метрики вверху — по всем строкам SKU. На **точечных и линейных** графиках данные "
            "сведены **по календарным дням**: цены — среднее по SKU, продажи — сумма по SKU."
        )

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
    c3.metric("Прибыль", f"{prod_df['profit'].sum():,.0f} ₽")

    if selected_product == "Все товары" and "store" in prod_df.columns:
        st.subheader("Рейтинг магазинов по прибыли")
        agg_map = {"profit": "sum", "revenue": "sum", "sales": "sum"}
        if "is_oos" in prod_df.columns:
            agg_map["is_oos"] = "sum"
        top_stores = prod_df.groupby("store", as_index=False).agg(agg_map).sort_values("profit", ascending=False)
        if "is_oos" in top_stores.columns:
            top_stores = top_stores.rename(columns={"is_oos": "oos_days"})
        top_stores["profit"] = top_stores["profit"].round(2)
        top_stores["revenue"] = top_stores["revenue"].round(2)
        st.dataframe(top_stores, width='stretch', hide_index=True)

    st.subheader("Зависимость продаж от нашей цены")
    kind1 = st.selectbox("Тип графика", CHART_LABELS, key="ov_chart_price_sales")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_price_vs_sales(ax, prod_df, kind1, aggregate_daily=(selected_product == "Все товары"))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Динамика нашей цены и цены конкурента")
    kind2 = st.selectbox("Тип графика", CHART_LABELS_TIME, key="ov_chart_time")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_prices_over_time(ax2, prod_df, kind2, aggregate_daily=(selected_product == "Все товары"))
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

    if selected_product == "Все товары":
        work_df = (
            prod_df.groupby("date", as_index=False)
            .agg(
                {
                    "our_price": "mean",
                    "competitor_price": "mean",
                    "sales": "sum",
                    "profit": "sum",
                    "cogs": "mean",
                }
            )
            .sort_values("date")
        )
        st.caption(
            "Для «всех товаров» показатели за день: **продажи и прибыль — сумма по SKU**, "
            "**цены — среднее по SKU**. Регрессия по цене и симуляция с методом regression "
            "считаются **отдельно по каждому SKU** (как в симуляции); ниже — сводка и агрегированный график."
        )
    else:
        work_df = prod_df.sort_values("date")

    last_row = work_df.iloc[-1]

    if selected_product == "Все товары":
        res = get_recommendations_all_products(prod_df, work_df)
        if not res:
            st.warning("В данных нет ни одного товара из каталога.")
            st.stop()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Текущее состояние (день)")
            st.metric("Средняя наша цена", f"{last_row['our_price']:.2f} ₽")
            st.metric("Средняя цена конкурента", f"{last_row['competitor_price']:.2f} ₽")
            st.metric("Суммарная прибыль", f"{last_row['profit']:.2f} ₽")
        with c2:
            st.subheader("Эвристика")
            st.metric("Средняя цена (правила)", f"{res['mean_rec_rules']:.2f} ₽", f"{res['growth_rules_pct']:+.1f}% к сумме прибыли")
            st.caption("Правило считается **по каждому SKU**; прогноз — из PRODUCTS, затем сумма по товарам.")
        with c3:
            st.subheader("Регрессия")
            st.metric("Средняя цена (регр.)", f"{res['mean_opt_reg']:.2f} ₽", f"{res['growth_reg_pct']:+.1f}% к сумме прибыли")
            st.caption("Оптимум **по каждому SKU** на своей истории; % — относительно суммы фактической прибыли за день.")
            if not res['all_rel']:
                st.warning(
                    "У части товаров наклон регрессии не отрицательный — для них оптимальная цена условна "
                    "(см. колонку «Надёжн.» в таблице)."
                )

        st.divider()
        t1, t2 = st.tabs(["Эвристика по SKU", "Регрессия по SKU"])
        with t1:
            st.dataframe(pd.DataFrame(res['rule_rows']), width='stretch', hide_index=True)
        with t2:
            st.dataframe(pd.DataFrame(res['reg_rows']), width='stretch', hide_index=True)

        st.divider()
        st.subheader("Агрегированная прибыль vs средняя цена портфеля (по дням)")
        st.caption(
            "Кривая строится по дневным точкам: средняя цена — ось X, суммарные продажи — регрессия; "
            "вершина — условный оптимум **для портфельного ряда**, не обязательно совпадает со средним "
            "пер-товарных оптимумов из таблицы."
        )
        anchor = res['opt_agg'] if res['opt_agg'] > 0 else float(last_row["our_price"])
        p_min = max(1.0, anchor * 0.5)
        p_max = max(p_min * 1.01, anchor * 1.5)
        prices = np.linspace(p_min, p_max, 100)
        revenues = prices * (res['a_agg'] - res['b_agg'] * prices)
        c_val = float(work_df['cogs'].mean()) if 'cogs' in work_df.columns else 0.0
        profits = (prices - c_val) * (res['a_agg'] - res['b_agg'] * prices)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prices, profits, label="Прогноз прибыли (агрегат)", color="gray", alpha=0.5)
        ax.axvline(last_row["our_price"], color="red", ls="--", label=f"Средняя сейчас ({last_row['our_price']:.2f})")
        if res['rel_agg'] and res['b_agg'] > 0:
            ax.axvline(res['opt_agg'], color="green", ls="-", label=f"Оптимум агрег. ({res['opt_agg']:.2f})")
        ax.set_xlabel("Средняя цена портфеля (₽)")
        ax.set_ylabel("Прогнозируемая прибыль (модель по дням)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"👉 **Итог по портфелю**: суммарная прибыль последнего дня {res['total_profit_actual']:.2f} ₽ → "
            f"суммарный прогноз по регрессии {res['total_pred_reg']:.2f} ₽ (**{res['growth_reg_pct']:+.1f}%**); "
            f"средняя рекомендованная цена по SKU {res['mean_opt_reg']:.2f} ₽."
        )
        return

    res = get_recommendations_single_product(prod_df, work_df, selected_product)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Текущее состояние")
        st.metric("Наша цена", f"{res['last_row']['our_price']:.2f} ₽")
        st.metric("Цена конкурента", f"{res['last_row']['competitor_price']:.2f} ₽")
        st.metric("Прибыль (день)", f"{res['last_row']['profit']:.2f} ₽")
    with c2:
        st.subheader("Эвристика")
        st.metric("Новая цена", f"{res['rec_price_rules']:.2f} ₽", f"{res['fc_rules']['growth_pct']:+.1f}% прибыли")
        st.caption(f"Правило: {res['rule_name']}")
    with c3:
        st.subheader("Регрессия")
        st.metric("Новая цена", f"{res['opt_price_reg']:.2f} ₽", f"{res['fc_reg']['growth_pct']:+.1f}% прибыли")
        st.caption(f"Формула: Profit = (P - C) * ({res['a']:.1f} - {res['b']:.2f}*P)")
        if not res['is_reg_reliable']:
            st.warning(
                "Наклон регрессии не отрицательный: оценка эластичности ненадежна, "
                "поэтому оптимальная цена может быть неточной."
            )

    st.divider()
    st.subheader("Анализ эластичности прибыли (Regression)")
    anchor = res['opt_price_reg'] if res['opt_price_reg'] > 0 else float(res['last_row']["our_price"])
    p_min = max(1.0, anchor * 0.5)
    p_max = max(p_min * 1.01, anchor * 1.5)
    prices = np.linspace(p_min, p_max, 100)
    
    comp_cogs = float(res['last_row'].get('cogs', PRODUCTS.get(selected_product, {}).get('cogs', 0.0)))
    profits = (prices - comp_cogs) * (res['a'] - res['b'] * prices)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices, profits, label="Прогноз прибыли", color="gray", alpha=0.5)
    ax.axvline(res['last_row']["our_price"], color="red", ls="--", label=f"Текущая ({res['last_row']['our_price']:.2f})")
    ax.axvline(res['opt_price_reg'], color="green", ls="-", label=f"Оптимальная ({res['opt_price_reg']:.2f})")
    ax.set_xlabel("Цена (₽)")
    ax.set_ylabel("Прогнозируемая прибыль")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"👉 **Итог по {selected_product}**: старая цена {res['last_row']['our_price']:.2f} ₽ → "
        f"новая цена {res['opt_price_reg']:.2f} ₽ | прогноз прибыли: {res['fc_reg']['growth_pct']:+.1f}%"
    )


def render_simulation_tab(df: pd.DataFrame, selected_product: str) -> None:
    st.title("🔮 Симуляция будущего")
    st.caption(
        f"Товар для прогноза: **{selected_product}** — выбирается только в боковой панели «Выберите товар»."
    )
    rs = st.session_state.get("ov_range_start")
    re = st.session_state.get("ov_range_end")
    if rs:
        period_text = f"**{rs}** — **{re if re else '...'}**"
        st.info(
            f"💡 Модели будут обучаться на историческом периоде: {period_text}. "
            "Вы можете изменить его в календаре на вкладке '📊 Обзор'."
        )

    col1, col2 = st.columns(2)
    n_steps = col1.slider("Горизонт симуляции (дней)", 7, 30, 14)
    method = col2.selectbox("Метод принятия решений", ["regression", "rules"])

    if st.button("Запустить симуляцию", type="primary"):
        _, prod_df_period = get_product_df_with_period(df, selected_product)
        if len(prod_df_period) < 2:
            st.error("⛔ Слишком короткий период для обучения. Выберите диапазон пошире в календаре.")
            st.stop()

        with st.spinner("Рынок просчитывается..."):
            simulated_df = simulate(
                prod_df_period, n_steps, method, target_product=selected_product
            )

        if selected_product == "Все товары":
            hist_rev = prod_df_period.groupby("date")["profit"].sum()
            sim_rev = simulated_df.groupby("date")["profit"].sum()
            title_suffix = "по всем товарам"
        else:
            hist_rev = prod_df_period[prod_df_period["product"] == selected_product].groupby("date")[
                "profit"
            ].sum()
            sim_rev = simulated_df[simulated_df["product"] == selected_product].groupby("date")[
                "profit"
            ].sum()
            title_suffix = f"по товару '{selected_product}'"

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
            "sim_scope": selected_product,
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
        and sl["sim_scope"] == selected_product
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

    if selected_product != "Все товары":
        pred_df = pred_df[pred_df["product"] == selected_product].copy()
    if pred_df.empty:
        st.info("В predict_sales.csv нет строк для выбранной области симуляции.")
        return

    scope_key = f"{selected_product}|{sl['n_steps']}|{sl['method']}"
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
