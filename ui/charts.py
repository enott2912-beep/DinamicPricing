import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

CHART_LABELS = ["Точечный", "Линейный", "Столбчатая диаграмма", "Гистограмма"]
CHART_LABELS_TIME = ["Точечный", "Линейный", "Столбчатая диаграмма", "Гистограмма"]


def render_stored_simulation(sim_last: dict) -> None:
    hist_rev = sim_last["hist_rev"]
    future_sim = sim_last["future_sim"]
    max_period_date = sim_last["max_period_date"]
    title_suffix = sim_last["title_suffix"]
    n_steps = sim_last["n_steps"]
    method = sim_last["method"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_rev.index, hist_rev.values, label="Выбранная история", color="blue", linewidth=1.5)
    if not future_sim.empty:
        ax.plot(
            future_sim.index,
            future_sim.values,
            label=f"Прогноз ({method})",
            color="green",
            ls="--",
            linewidth=2,
        )
    ax.axvline(max_period_date, color="black", alpha=0.3, linestyle=":")
    ax.set_ylabel(f"Прибыль {title_suffix} (₽)")
    ax.set_title(f"Сценарный прогноз на {n_steps} дней (от {pd.Timestamp(max_period_date).date()})")
    ax.legend()
    
    # Продвинутое форматирование оси дат для огромных интервалов (30-100+ дней)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45, ha='right')
    
    st.pyplot(fig)
    plt.close(fig)

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


def plot_price_vs_sales(ax, prod_df: pd.DataFrame, kind: str, *, aggregate_daily: bool = False) -> None:
    if len(prod_df) == 0:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
        return

    use_daily = aggregate_daily and kind in ("Точечный", "Линейный")
    if use_daily:
        work = _daily_price_sales_agg(prod_df)
        x = work["our_price"].values
        y = work["sales"].values
        if len(work) == 0:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
            return
    else:
        x = prod_df["our_price"].values
        y = prod_df["sales"].values

    if kind == "Точечный":
        ax.scatter(x, y, alpha=0.6, c="#1f77b4")
        ax.set_xlabel("Цена (₽)" + (" — средняя по SKU за день" if use_daily else ""))
        ax.set_ylabel("Продажи (шт)" + (" — сумма по SKU за день" if use_daily else ""))
    elif kind == "Линейный":
        # По оси X — цена: соединяем точки в порядке возрастания цены (как у одного SKU),
        # иначе при дневной агрегации хронологический порядок даёт хаотичную «змейку».
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker=".", alpha=0.8, color="#1f77b4")
        ax.set_xlabel("Цена (₽)" + (" — средняя по SKU за день" if use_daily else ""))
        ax.set_ylabel("Продажи (шт)" + (" — сумма по SKU за день" if use_daily else ""))
    elif kind == "Столбчатая диаграмма":
        unique_prices = np.sort(prod_df["our_price"].unique())
        if len(unique_prices) > 10:
            bins = np.linspace(unique_prices.min(), unique_prices.max(), 9)
            labels = [f"{bins[i]:.0f}-{bins[i + 1]:.0f}" for i in range(len(bins) - 1)]
            work = prod_df.copy()
            work["price_bin"] = pd.cut(work["our_price"], bins=bins, labels=labels, include_lowest=True)
            grouped = work.groupby("price_bin", as_index=False)["sales"].sum()
            ax.bar(grouped["price_bin"].astype(str), grouped["sales"], color="#1f77b4", alpha=0.85)
            ax.set_xlabel("Диапазон цены (₽)")
        else:
            grouped = prod_df.groupby("our_price", as_index=False)["sales"].sum().sort_values("our_price")
            ax.bar(grouped["our_price"].astype(str), grouped["sales"], color="#1f77b4", alpha=0.85)
            ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Продажи (шт)")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Гистограмма":
        bins = min(20, max(5, len(prod_df) // 3))
        ax.hist(x, bins=bins, weights=y, color="#1f77b4", alpha=0.75, edgecolor="white")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Сумма продаж (вес по дням)")

    ax.grid(True, alpha=0.3)


def plot_prices_over_time(ax, prod_df: pd.DataFrame, kind: str, *, aggregate_daily: bool = False) -> None:
    if len(prod_df) == 0:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
        return

    use_daily = aggregate_daily and kind in ("Точечный", "Линейный")
    if use_daily:
        plot_df = _daily_prices_agg(prod_df)
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
            return
        t = plot_df["date"].values
        p1 = plot_df["our_price"].values
        p2 = plot_df["competitor_1_price"].values
        p3 = plot_df["competitor_2_price"].values
    else:
        t = prod_df["date"].values
        p1 = prod_df["our_price"].values
        p2_col = "competitor_1_price" if "competitor_1_price" in prod_df.columns else "competitor_price"
        p3_col = "competitor_2_price" if "competitor_2_price" in prod_df.columns else "competitor_price"
        p2 = prod_df[p2_col].values
        p3 = prod_df[p3_col].values

    if kind == "Точечный":
        l1, l2, l3 = (
            ("Наша цена (средняя по SKU за день)", "Конкурент 1 (средняя)", "Конкурент 2 (средняя)")
            if use_daily
            else ("Наша цена", "Цена конкурента 1", "Цена конкурента 2")
        )
        ax.scatter(t, p1, label=l1, alpha=0.8, c="#1f77b4")
        ax.scatter(t, p2, label=l2, alpha=0.6, c="#ff7f0e", marker="s")
        ax.scatter(t, p3, label=l3, alpha=0.6, c="#2ca02c", marker="^")
        ax.set_ylabel("Цена (₽)" + (" — средняя по SKU за день" if use_daily else ""))
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Линейный":
        l1, l2, l3 = (
            ("Наша цена (средняя по SKU за день)", "Конкурент 1 (средняя)", "Конкурент 2 (средняя)")
            if use_daily
            else ("Наша цена", "Цена конкурента 1", "Цена конкурента 2")
        )
        ax.plot(t, p1, label=l1, marker=".", alpha=0.8)
        ax.plot(t, p2, label=l2, ls="--", alpha=0.7)
        ax.plot(t, p3, label=l3, ls=":", alpha=0.8)
        ax.set_ylabel("Цена (₽)" + (" — средняя по SKU за день" if use_daily else ""))
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Столбчатая диаграмма":
        n = len(prod_df)
        if n > 30:
            df_copy = prod_df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"])
            days_range = (df_copy["date"].max() - df_copy["date"].min()).days
            if days_range <= 60:
                freq, freq_name, label_fmt = "3D", "по 3 дня", "%m-%d"
            elif days_range <= 180:
                freq, freq_name, label_fmt = "W", "по неделям", "%m-%d"
            else:
                freq, freq_name, label_fmt = "ME", "по месяцам", "%Y-%m"
            agg_map = {"our_price": "mean", "competitor_price": "mean"}
            if "competitor_1_price" in df_copy.columns:
                agg_map["competitor_1_price"] = "mean"
            if "competitor_2_price" in df_copy.columns:
                agg_map["competitor_2_price"] = "mean"
            df_agg = df_copy.groupby(pd.Grouper(key="date", freq=freq)).agg(agg_map).dropna()
            if len(df_agg) == 0:
                ax.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center", transform=ax.transAxes)
                return
            x = np.arange(len(df_agg))
            p1_agg = df_agg["our_price"].values
            p2_col = "competitor_1_price" if "competitor_1_price" in df_agg.columns else "competitor_price"
            p3_col = "competitor_2_price" if "competitor_2_price" in df_agg.columns else "competitor_price"
            p2_agg = df_agg[p2_col].values
            p3_agg = df_agg[p3_col].values
            date_labels = [d.strftime(label_fmt) for d in df_agg.index]
            ax.bar(x - 0.25, p1_agg, width=0.25, label="Наша цена (средняя)", alpha=0.8)
            ax.bar(x, p2_agg, width=0.25, label="Конкурент 1 (средний)", alpha=0.8)
            ax.bar(x + 0.25, p3_agg, width=0.25, label="Конкурент 2 (средний)", alpha=0.8)
            ax.set_title(f"Средние цены (сгруппировано {freq_name})", fontsize=9)
        else:
            x = np.arange(n)
            w = 0.25
            ax.bar(x - w, p1, width=w, label="Наша цена", alpha=0.8)
            ax.bar(x, p2, width=w, label="Конкурент 1", alpha=0.8)
            ax.bar(x + w, p3, width=w, label="Конкурент 2", alpha=0.8)
            date_labels = [pd.Timestamp(ti).strftime("%m-%d") for ti in t]

        ax.set_xticks(x)
        if len(date_labels) > 15:
            step = max(1, len(date_labels) // 12)
            visible_labels = [date_labels[i] if i % step == 0 else "" for i in range(len(date_labels))]
            ax.set_xticklabels(visible_labels, rotation=45, ha="right", fontsize=8)
            ax.set_xticks(list(range(0, len(date_labels), step)))
        else:
            ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Цена (₽)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
    elif kind == "Гистограмма":
        bins = min(15, max(5, len(prod_df) // 2))
        ax.hist(p1, bins=bins, alpha=0.7, label="Наша цена", color="#1f77b4")
        ax.hist(p2, bins=bins, alpha=0.5, label="Конкурент 1", color="#ff7f0e")
        ax.hist(p3, bins=bins, alpha=0.4, label="Конкурент 2", color="#2ca02c")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Число дней")
        ax.legend()
