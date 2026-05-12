import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

CHART_LABELS = ["Точечный", "Линейный", "Столбчатая диаграмма", "Гистограмма"]
CHART_LABELS_TIME = ["Точечный", "Линейный", "Столбчатая диаграмма", "Гистограмма"]


def render_stored_simulation(sim_last: dict) -> None:
    hist_rev = sim_last["hist_rev"]
    future_sim = sim_last["future_sim"]
    max_period_date = sim_last["max_period_date"]
    title_suffix = sim_last["title_suffix"]
    n_steps = sim_last["n_steps"]
    method = sim_last["method"]

    fig = go.Figure()

    # Историческая прибыль
    fig.add_trace(go.Scatter(
        x=hist_rev.index, 
        y=hist_rev.values, 
        mode='lines', 
        name="Выбранная история",
        line=dict(color='blue', width=2)
    ))

    # Прогнозная прибыль
    if not future_sim.empty:
        fig.add_trace(go.Scatter(
            x=future_sim.index, 
            y=future_sim.values, 
            mode='lines', 
            name=f"Прогноз ({method})",
            line=dict(color='green', width=3, dash='dash')
        ))

    # Вертикальная линия окончания периода истории
    fig.add_vline(x=pd.Timestamp(max_period_date).timestamp() * 1000, line_width=2, line_dash="dash", line_color="black", opacity=0.3)

    fig.update_layout(
        title=f"Сценарный прогноз на {n_steps} дней (от {pd.Timestamp(max_period_date).date()})",
        yaxis_title=f"Прибыль {title_suffix} (₽)",
        xaxis_title="Дата",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')

    avg_sim = sim_last["avg_sim"]
    delta = sim_last["delta"]
    first_day_rev = sim_last["first_day_rev"]
    last_day_rev = sim_last["last_day_rev"]
    future_empty = sim_last["future_empty"]

    m1, m2, m3 = st.columns(3)
    if future_empty or (avg_sim is not None and np.isnan(avg_sim)):
        m1.warning(
            "Не удалось вычислить ожидаемую прибыль: в симуляции нет корректных будущих точек "
            "относительно выбранного периода."
        )
        m2.metric("Ожидаемая прибыль (1-й день)", "—")
        m3.metric("Ожидаемая прибыль (последний день)", "—")
    else:
        m1.metric(
            "Ожидаемая прибыль/день",
            f"{avg_sim:,.0f} ₽",
            f"{delta:+.1f}%",
            help="Среднее значение прибыли за период симуляции в сравнении с историческим средним.",
        )
        m2.metric("Ожидаемая прибыль (1-й день)", f"{first_day_rev:,.0f} ₽")
        m3.metric("Ожидаемая прибыль (последний день)", f"{last_day_rev:,.0f} ₽")


def _daily_price_sales_agg(prod_df: pd.DataFrame) -> pd.DataFrame:
    tmp = prod_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    return (
        tmp.groupby("date", as_index=False)
        .agg(our_price=("our_price", "mean"), sales=("sales", "sum"))
        .sort_values("date")
    )


def _daily_prices_agg(prod_df: pd.DataFrame) -> pd.DataFrame:
    tmp = prod_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    agg_map = {"our_price": "mean", "competitor_price": "mean"}
    if "competitor_1_price" in tmp.columns:
        agg_map["competitor_1_price"] = "mean"
    if "competitor_2_price" in tmp.columns:
        agg_map["competitor_2_price"] = "mean"
    out = tmp.groupby("date", as_index=False).agg(agg_map).sort_values("date")
    if "competitor_1_price" not in out.columns:
        out["competitor_1_price"] = out["competitor_price"]
    if "competitor_2_price" not in out.columns:
        out["competitor_2_price"] = out["competitor_price"]
    return out


def plot_price_vs_sales(prod_df: pd.DataFrame, kind: str, *, aggregate_daily: bool = False) -> go.Figure:
    fig = go.Figure()
    if len(prod_df) == 0:
        fig.update_layout(title="Нет данных", xaxis_visible=False, yaxis_visible=False)
        return fig

    use_daily = aggregate_daily and kind in ("Точечный", "Линейный")
    if use_daily:
        work = _daily_price_sales_agg(prod_df)
        if len(work) == 0:
            fig.update_layout(title="Нет данных", xaxis_visible=False, yaxis_visible=False)
            return fig
    else:
        work = prod_df.copy()

    x_labels = "Цена (₽)" + (" — средняя по SKU за день" if use_daily else "")
    y_labels = "Продажи (шт)"

    if kind == "Точечный":
        y_labels += " — сумма по SKU за день" if use_daily else ""
        fig = px.scatter(work, x="our_price", y="sales", opacity=0.7, color_discrete_sequence=["#1f77b4"])
        fig.update_layout(xaxis_title=x_labels, yaxis_title=y_labels)

    elif kind == "Линейный":
        # Group by price to prevent zigzag lines, finding the mean sales per price point
        grouped = work.groupby("our_price", as_index=False)["sales"].mean().sort_values("our_price")
        y_labels += " (усреднено для каждой цены)"
        fig = px.line(grouped, x="our_price", y="sales", markers=True, color_discrete_sequence=["#1f77b4"])
        fig.update_layout(xaxis_title=x_labels, yaxis_title=y_labels)

    elif kind == "Столбчатая диаграмма":
        unique_prices = np.sort(prod_df["our_price"].unique())
        if len(unique_prices) > 10:
            bins = np.linspace(unique_prices.min(), unique_prices.max(), 9)
            labels = [f"{bins[i]:.0f}-{bins[i + 1]:.0f}" for i in range(len(bins) - 1)]
            w = prod_df.copy()
            w["price_bin"] = pd.cut(w["our_price"], bins=bins, labels=labels, include_lowest=True)
            grouped = w.groupby("price_bin", as_index=False, observed=False)["sales"].sum()
            fig = px.bar(grouped, x=grouped["price_bin"].astype(str), y="sales", color_discrete_sequence=["#1f77b4"])
            fig.update_layout(xaxis_title="Диапазон цены (₽)", yaxis_title="Продажи (шт)")
        else:
            grouped = prod_df.groupby("our_price", as_index=False)["sales"].sum().sort_values("our_price")
            fig = px.bar(grouped, x=grouped["our_price"].astype(str), y="sales", color_discrete_sequence=["#1f77b4"])
            fig.update_layout(xaxis_title="Цена (₽)", yaxis_title="Продажи (шт)")

    elif kind == "Гистограмма":
        bins = min(20, max(5, len(prod_df) // 3))
        # Имитируем взвешенную гистограмму, но px.histogram умеет y=sales
        fig = px.histogram(work, x="our_price", y="sales", nbins=bins, color_discrete_sequence=["#1f77b4"])
        fig.update_layout(xaxis_title="Цена (₽)", yaxis_title="Сумма продаж (вес по дням)")

    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    return fig


def plot_prices_over_time(prod_df: pd.DataFrame, kind: str, *, aggregate_daily: bool = False) -> go.Figure:
    fig = go.Figure()
    if len(prod_df) == 0:
        fig.update_layout(title="Нет данных", xaxis_visible=False, yaxis_visible=False)
        return fig

    use_daily = aggregate_daily and kind in ("Точечный", "Линейный")
    if use_daily:
        plot_df = _daily_prices_agg(prod_df)
        if len(plot_df) == 0:
            fig.update_layout(title="Нет данных", xaxis_visible=False, yaxis_visible=False)
            return fig
    else:
        plot_df = prod_df.copy()
        if "competitor_1_price" not in plot_df.columns:
            plot_df["competitor_1_price"] = plot_df["competitor_price"]
        if "competitor_2_price" not in plot_df.columns:
            plot_df["competitor_2_price"] = plot_df["competitor_price"]

    l1, l2, l3 = (
        ("Наша цена (средняя)", "Конкурент 1 (средняя)", "Конкурент 2 (средняя)")
        if use_daily else ("Наша цена", "Цена конкурента 1", "Цена конкурента 2")
    )
    y_label = "Цена (₽)" + (" — средняя по SKU за день" if use_daily else "")

    if kind == "Точечный":
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["our_price"], mode='markers', name=l1, marker=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["competitor_1_price"], mode='markers', name=l2, marker=dict(color="#ff7f0e", symbol="square")))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["competitor_2_price"], mode='markers', name=l3, marker=dict(color="#2ca02c", symbol="triangle-up")))
        fig.update_layout(yaxis_title=y_label)

    elif kind == "Линейный":
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["our_price"], mode='lines+markers', name=l1, line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["competitor_1_price"], mode='lines', name=l2, line=dict(color="#ff7f0e", dash="dash")))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["competitor_2_price"], mode='lines', name=l3, line=dict(color="#2ca02c", dash="dot")))
        fig.update_layout(yaxis_title=y_label)

    elif kind == "Столбчатая диаграмма":
        n = len(prod_df)
        if n > 30:
            df_copy = prod_df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"])
            days_range = (df_copy["date"].max() - df_copy["date"].min()).days
            if days_range <= 60:
                freq = "3D"
            elif days_range <= 180:
                freq = "W"
            else:
                freq = "ME"
            agg_map = {"our_price": "mean", "competitor_price": "mean"}
            if "competitor_1_price" in df_copy.columns:
                agg_map["competitor_1_price"] = "mean"
            if "competitor_2_price" in df_copy.columns:
                agg_map["competitor_2_price"] = "mean"
            df_agg = df_copy.groupby(pd.Grouper(key="date", freq=freq)).agg(agg_map).dropna()
            
            p2_col = "competitor_1_price" if "competitor_1_price" in df_agg.columns else "competitor_price"
            p3_col = "competitor_2_price" if "competitor_2_price" in df_agg.columns else "competitor_price"

            fig.add_trace(go.Bar(x=df_agg.index, y=df_agg["our_price"], name=l1, marker_color="#1f77b4"))
            fig.add_trace(go.Bar(x=df_agg.index, y=df_agg[p2_col], name=l2, marker_color="#ff7f0e"))
            fig.add_trace(go.Bar(x=df_agg.index, y=df_agg[p3_col], name=l3, marker_color="#2ca02c"))
            fig.update_layout(barmode='group', title=f"Средние цены (группировано)")
        else:
            fig.add_trace(go.Bar(x=plot_df["date"], y=plot_df["our_price"], name=l1, marker_color="#1f77b4"))
            fig.add_trace(go.Bar(x=plot_df["date"], y=plot_df["competitor_1_price"], name=l2, marker_color="#ff7f0e"))
            fig.add_trace(go.Bar(x=plot_df["date"], y=plot_df["competitor_2_price"], name=l3, marker_color="#2ca02c"))
            fig.update_layout(barmode='group')
        fig.update_layout(yaxis_title="Цена (₽)")

    elif kind == "Гистограмма":
        # Поскольку px.histogram не может рисовать поверх легко, используем go.Histogram
        fig.add_trace(go.Histogram(x=plot_df["our_price"], name=l1, opacity=0.7, marker_color="#1f77b4"))
        fig.add_trace(go.Histogram(x=plot_df["competitor_1_price"], name=l2, opacity=0.5, marker_color="#ff7f0e"))
        fig.add_trace(go.Histogram(x=plot_df["competitor_2_price"], name=l3, opacity=0.4, marker_color="#2ca02c"))
        fig.update_layout(barmode='overlay', xaxis_title="Цена (₽)", yaxis_title="Число дней")

    fig.update_layout(hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
