import pandas as pd
import numpy as np

from model.pricing import (
    apply_rules,
    fit_regression,
    fit_regression_aggregate_daily,
    forecast,
    products_in_dataframe,
    recommend_price_lightgbm,
)


def _exclude_oos_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "is_oos" in df.columns:
        return df[~df["is_oos"].astype(bool)].copy()
    return df.copy()


def _stable_profit_baseline(df: pd.DataFrame) -> float:
    """
    Устойчивый baseline прибыли: медиана последних валидных дней.
    Это защищает % роста от всплесков при OOS/нулевом последнем дне.
    """
    clean = _exclude_oos_rows(df)
    if clean.empty:
        return 0.0
    tail = clean["profit"].tail(7)
    if tail.empty:
        return 0.0
    return float(tail.median())


def get_recommendations_all_products(prod_df: pd.DataFrame, work_df: pd.DataFrame, include_lightgbm: bool = True) -> dict:
    """
    Бизнес-логика расчета рекомендаций для портфеля (SRP: отделено от UI).
    """
    sku_list = products_in_dataframe(prod_df)
    if not sku_list:
        return {}

    rule_rows = []
    reg_rows = []
    nl_rows = []
    rec_prices_rules = []
    opts = []
    nl_opts = []
    rel_flags = []
    nl_rel_flags = []
    total_profit_actual = 0.0
    total_pred_rules = 0.0
    total_pred_reg = 0.0
    total_pred_nl = 0.0

    for p in sku_list:
        sub = prod_df[prod_df["product"] == p].sort_values("date")
        if sub.empty:
            continue

        sub_valid = _exclude_oos_rows(sub)
        if sub_valid.empty:
            continue
        last_i = sub_valid.iloc[-1]
        prev = sub_valid["sales"].iloc[:-1]
        avg7 = float(prev.tail(7).mean()) if len(prev) > 0 else float(last_i["sales"])
        current_profit = _stable_profit_baseline(sub_valid)
        rp, rn = apply_rules(last_i, avg7)

        fc_r = forecast(p, rp, current_profit)
        total_profit_actual += current_profit
        total_pred_rules += float(fc_r["forecast_profit"])
        rec_prices_rules.append(rp)

        rule_rows.append({
            "Товар": p,
            "Цена, ₽": round(float(last_i["our_price"]), 2),
            "Цена (правила), ₽": rp,
            "Правило": rn,
            "Прогноз прибыли, ₽": fc_r["forecast_profit"],
            "Δ к факту, %": fc_r["growth_pct"],
        })

        a_i, b_i, opt_i, rel_i = fit_regression(prod_df, p)
        fc_g = forecast(p, opt_i, current_profit, regression_params=(a_i, b_i))

        total_pred_reg += float(fc_g["forecast_profit"])
        opts.append(opt_i)
        rel_flags.append(rel_i)

        reg_rows.append({
            "Товар": p,
            "Цена, ₽": round(float(last_i["our_price"]), 2),
            "Цена (регр.), ₽": opt_i,
            "Надёжн.": "да" if rel_i else "нет",
            "Прогноз прибыли, ₽": fc_g["forecast_profit"],
            "Δ к факту, %": fc_g["growth_pct"],
            "A": round(a_i, 2),
            "B": round(b_i, 4),
        })

        if include_lightgbm:
            cogs_i = float(last_i.get("cogs", 0.0))
            nl_res = recommend_price_lightgbm(
                history_df=sub_valid,
                next_date=pd.Timestamp(last_i["date"]) + pd.Timedelta(days=1),
                last_price=float(last_i["our_price"]),
                competitor_price=float(last_i.get("competitor_price", last_i["our_price"])),
                cogs=cogs_i,
            )
            pred_profit_nl = float(nl_res["pred_sales"]) * (float(nl_res["recommended_price"]) - cogs_i)
            total_pred_nl += pred_profit_nl
            nl_opts.append(float(nl_res["recommended_price"]))
            nl_rel_flags.append(bool(nl_res["reliable"]))
            nl_rows.append({
                "Товар": p,
                "Цена, ₽": round(float(last_i["our_price"]), 2),
                "Цена (LightGBM), ₽": round(float(nl_res["recommended_price"]), 2),
                "Надёжн.": "да" if nl_res["reliable"] else "нет",
                "Прогноз прибыли, ₽": round(pred_profit_nl, 2),
                "Δ к факту, %": round(((pred_profit_nl - current_profit) / current_profit * 100) if current_profit > 0 else 0.0, 1),
                "Диагностика": "; ".join(nl_res["warnings"]) if nl_res["warnings"] else "OK",
            })

    a_agg, b_agg, opt_agg, rel_agg = fit_regression_aggregate_daily(work_df)

    return {
        "rule_rows": rule_rows,
        "reg_rows": reg_rows,
        "nl_rows": nl_rows,
        "mean_rec_rules": float(np.mean(rec_prices_rules)) if rec_prices_rules else 0.0,
        "mean_opt_reg": float(np.mean(opts)) if opts else 0.0,
        "mean_opt_nl": float(np.mean(nl_opts)) if nl_opts else 0.0,
        "growth_rules_pct": (
            (total_pred_rules - total_profit_actual) / total_profit_actual * 100
            if total_profit_actual > 0 else 0.0
        ),
        "growth_reg_pct": (
            (total_pred_reg - total_profit_actual) / total_profit_actual * 100
            if total_profit_actual > 0 else 0.0
        ),
        "total_profit_actual": total_profit_actual,
        "total_pred_reg": total_pred_reg,
        "total_pred_nl": total_pred_nl,
        "a_agg": a_agg,
        "b_agg": b_agg,
        "opt_agg": opt_agg,
        "rel_agg": rel_agg,
        "all_rel": all(rel_flags),
        "all_nl_rel": all(nl_rel_flags) if nl_rel_flags else False,
        "growth_nl_pct": (
            (total_pred_nl - total_profit_actual) / total_profit_actual * 100
            if total_profit_actual > 0 else 0.0
        ),
    }


def get_recommendations_single_product(
    prod_df: pd.DataFrame,
    work_df: pd.DataFrame,
    selected_product: str,
    include_lightgbm: bool = True,
) -> dict:
    """
    Бизнес-логика расчета рекомендаций для одного товара (SRP).
    """
    valid_df = _exclude_oos_rows(work_df)
    last_row = valid_df.iloc[-1] if not valid_df.empty else work_df.iloc[-1]
    prev_sales = valid_df["sales"].iloc[:-1] if len(valid_df) > 1 else work_df["sales"].iloc[:-1]
    avg7 = prev_sales.tail(7).mean() if len(prev_sales) > 0 else last_row["sales"]
    current_profit = _stable_profit_baseline(valid_df if not valid_df.empty else work_df)

    rec_price_rules, rule_name = apply_rules(last_row, avg7)
    fc_rules = forecast(selected_product, rec_price_rules, current_profit)

    a, b, opt_price_reg, is_reg_reliable = fit_regression(prod_df, selected_product)
    fc_reg = forecast(selected_product, opt_price_reg, current_profit, regression_params=(a, b))
    cogs = float(last_row.get("cogs", 0.0))
    if include_lightgbm:
        nl_res = recommend_price_lightgbm(
            history_df=(valid_df if not valid_df.empty else work_df),
            next_date=pd.Timestamp(last_row["date"]) + pd.Timedelta(days=1),
            last_price=float(last_row["our_price"]),
            competitor_price=float(last_row.get("competitor_price", last_row["our_price"])),
            cogs=cogs,
        )
    else:
        nl_res = {"recommended_price": float(last_row["our_price"]), "reliable": False, "warnings": ["Расчёт LightGBM отключён."], "pred_sales": float(last_row["sales"])}
    pred_profit_nl = float(nl_res["pred_sales"]) * (float(nl_res["recommended_price"]) - cogs)
    growth_nl = ((pred_profit_nl - current_profit) / current_profit * 100) if current_profit > 0 else 0.0

    return {
        "last_row": last_row,
        "rec_price_rules": rec_price_rules,
        "rule_name": rule_name,
        "fc_rules": fc_rules,
        "a": a,
        "b": b,
        "opt_price_reg": opt_price_reg,
        "is_reg_reliable": is_reg_reliable,
        "fc_reg": fc_reg,
        "opt_price_nl": float(nl_res["recommended_price"]),
        "is_nl_reliable": bool(nl_res["reliable"]),
        "nl_warnings": nl_res["warnings"],
        "fc_nl": {
            "forecast_profit": round(pred_profit_nl, 2),
            "growth_pct": round(growth_nl, 1),
        },
    }
