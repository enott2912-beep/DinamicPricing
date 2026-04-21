import pandas as pd
import numpy as np

from model.pricing import (
    PRODUCTS,
    apply_rules,
    fit_regression,
    fit_regression_aggregate_daily,
    forecast,
    products_in_dataframe,
)


def get_recommendations_all_products(prod_df: pd.DataFrame, work_df: pd.DataFrame) -> dict:
    """
    Бизнес-логика расчета рекомендаций для портфеля (SRP: отделено от UI).
    """
    sku_list = products_in_dataframe(prod_df)
    if not sku_list:
        return {}

    rule_rows = []
    reg_rows = []
    rec_prices_rules = []
    opts = []
    rel_flags = []
    total_profit_actual = 0.0
    total_pred_rules = 0.0
    total_pred_reg = 0.0

    for p in sku_list:
        sub = prod_df[prod_df["product"] == p].sort_values("date")
        if sub.empty:
            continue
            
        last_i = sub.iloc[-1]
        prev = sub["sales"].iloc[:-1]
        avg7 = float(prev.tail(7).mean()) if len(prev) > 0 else float(last_i["sales"])
        rp, rn = apply_rules(last_i, avg7)
        
        fc_r = forecast(p, rp, float(last_i["profit"]))
        total_profit_actual += float(last_i["profit"])
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
        fc_g = forecast(p, opt_i, float(last_i["profit"]), regression_params=(a_i, b_i))
        
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

    a_agg, b_agg, opt_agg, rel_agg = fit_regression_aggregate_daily(work_df)

    return {
        "rule_rows": rule_rows,
        "reg_rows": reg_rows,
        "mean_rec_rules": float(np.mean(rec_prices_rules)) if rec_prices_rules else 0.0,
        "mean_opt_reg": float(np.mean(opts)) if opts else 0.0,
        "growth_rules_pct": (total_pred_rules - total_profit_actual) / total_profit_actual * 100 if total_profit_actual > 0 else 0.0,
        "growth_reg_pct": (total_pred_reg - total_profit_actual) / total_profit_actual * 100 if total_profit_actual > 0 else 0.0,
        "total_profit_actual": total_profit_actual,
        "total_pred_reg": total_pred_reg,
        "a_agg": a_agg,
        "b_agg": b_agg,
        "opt_agg": opt_agg,
        "rel_agg": rel_agg,
        "all_rel": all(rel_flags)
    }


def get_recommendations_single_product(prod_df: pd.DataFrame, work_df: pd.DataFrame, selected_product: str) -> dict:
    """
    Бизнес-логика расчета рекомендаций для одного товара (SRP).
    """
    last_row = work_df.iloc[-1]
    prev_sales = work_df["sales"].iloc[:-1]
    avg7 = prev_sales.tail(7).mean() if len(prev_sales) > 0 else last_row["sales"]

    rec_price_rules, rule_name = apply_rules(last_row, avg7)
    fc_rules = forecast(selected_product, rec_price_rules, float(last_row["profit"]))

    a, b, opt_price_reg, is_reg_reliable = fit_regression(prod_df, selected_product)
    fc_reg = forecast(selected_product, opt_price_reg, float(last_row["profit"]), regression_params=(a, b))

    return {
        "last_row": last_row,
        "rec_price_rules": rec_price_rules,
        "rule_name": rule_name,
        "fc_rules": fc_rules,
        "a": a,
        "b": b,
        "opt_price_reg": opt_price_reg,
        "is_reg_reliable": is_reg_reliable,
        "fc_reg": fc_reg
    }
