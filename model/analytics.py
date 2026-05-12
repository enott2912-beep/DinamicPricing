import pandas as pd
import numpy as np
import logging

from model.pricing import (
    apply_rules,
    fit_regression,
    fit_regression_aggregate_daily,
    forecast,
    products_in_dataframe,
    recommend_price_lightgbm,
)
from model.utils import (
    safe_growth_pct,
    safe_mean_or_none,
    format_rule_names,
    safe_all_check,
)

logger = logging.getLogger(__name__)

ENTITY_KEYS = ("store_id", "store", "store_profile", "brand_id", "brand", "product_id")


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


def _get_entity_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ENTITY_KEYS if c in df.columns]


def _iter_entities(product_df: pd.DataFrame) -> list[pd.DataFrame]:
    entity_cols = _get_entity_cols(product_df)
    if not entity_cols:
        return [product_df.sort_values("date")]
    return [grp.sort_values("date") for _, grp in product_df.groupby(entity_cols, dropna=False, sort=False)]


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

        entity_frames = _iter_entities(sub)
        if not entity_frames:
            continue
        last_prices: list[float] = []
        rule_prices: list[float] = []
        rule_names: list[str] = []
        reg_prices: list[float] = []
        reg_rel_entity: list[bool] = []
        reg_a_vals: list[float] = []
        reg_b_vals: list[float] = []
        nl_prices: list[float] = []
        nl_rel_entity: list[bool] = []
        nl_warns: list[str] = []
        cur_profit_sum = 0.0
        pred_rules_sum = 0.0
        pred_reg_sum = 0.0
        pred_nl_sum = 0.0

        for ent_df in entity_frames:
            ent_valid = _exclude_oos_rows(ent_df)
            if ent_valid.empty:
                logger.debug(f"Пропущена сущность товара '{p}' из-за отсутствия валидных данных (все OOS или нулевые продажи)")
                continue
            last_i = ent_valid.iloc[-1]
            prev = ent_valid["sales"].iloc[:-1]
            avg7 = float(prev.tail(7).mean()) if len(prev) > 0 else float(last_i["sales"])
            current_profit_ent = _stable_profit_baseline(ent_valid)
            cur_profit_sum += current_profit_ent
            last_prices.append(float(last_i["our_price"]))

            rp, rn = apply_rules(last_i, avg7)
            fc_r = forecast(p, rp, current_profit_ent)
            rule_prices.append(float(rp))
            rule_names.append(rn)
            pred_rules_sum += float(fc_r["forecast_profit"])

            a_i, b_i, opt_i, rel_i = fit_regression(ent_valid, p)
            fc_g = forecast(p, opt_i, current_profit_ent, regression_params=(a_i, b_i))
            reg_prices.append(float(opt_i))
            reg_rel_entity.append(bool(rel_i))
            reg_a_vals.append(float(a_i))
            reg_b_vals.append(float(b_i))
            pred_reg_sum += float(fc_g["forecast_profit"])

            if include_lightgbm:
                cogs_i = float(last_i.get("cogs", 0.0))
                nl_res = recommend_price_lightgbm(
                    history_df=ent_valid,
                    next_date=pd.Timestamp(last_i["date"]) + pd.Timedelta(days=1),
                    last_price=float(last_i["our_price"]),
                    competitor_price=float(last_i.get("competitor_price", last_i["our_price"])),
                    cogs=cogs_i,
                )
                pred_profit_nl = float(nl_res["pred_sales"]) * (float(nl_res["recommended_price"]) - cogs_i)
                pred_nl_sum += pred_profit_nl
                nl_prices.append(float(nl_res["recommended_price"]))
                nl_rel_entity.append(bool(nl_res["reliable"]))
                nl_warns.extend(list(nl_res["warnings"]))

        if not last_prices:
            continue

        total_profit_actual += cur_profit_sum
        total_pred_rules += pred_rules_sum
        total_pred_reg += pred_reg_sum
        rec_prices_rules.append(safe_mean_or_none(rule_prices))
        opts.append(safe_mean_or_none(reg_prices))
        rel_flags.append(safe_all_check(reg_rel_entity))

        rule_name = format_rule_names(rule_names)
        growth_rules_pct = safe_growth_pct(pred_rules_sum, cur_profit_sum)
        growth_reg_pct = safe_growth_pct(pred_reg_sum, cur_profit_sum)
        growth_nl_pct = safe_growth_pct(pred_nl_sum, cur_profit_sum)

        rule_rows.append({
            "Товар": p,
            "Цена, ₽": round(float(np.mean(last_prices)), 2),
            "Цена (правила), ₽": round(safe_mean_or_none(rule_prices) or 0.0, 2),
            "Правило": rule_name,
            "Прогноз прибыли, ₽": round(pred_rules_sum, 2),
            "Δ к факту, %": round(growth_rules_pct, 1),
        })

        reg_rows.append({
            "Товар": p,
            "Цена, ₽": round(float(np.mean(last_prices)), 2),
            "Цена (регр.), ₽": round(safe_mean_or_none(reg_prices) or 0.0, 2),
            "Надёжн.": "да" if safe_all_check(reg_rel_entity) else "нет",
            "Прогноз прибыли, ₽": round(pred_reg_sum, 2),
            "Δ к факту, %": round(growth_reg_pct, 1),
            "A": round(safe_mean_or_none(reg_a_vals) or 0.0, 2),
            "B": round(safe_mean_or_none(reg_b_vals) or 0.0, 4),
        })

        if include_lightgbm:
            total_pred_nl += pred_nl_sum
            nl_opts.append(safe_mean_or_none(nl_prices))
            nl_rel_flags.append(safe_all_check(nl_rel_entity))
            warn_unique = list(dict.fromkeys([w for w in nl_warns if w]))
            nl_rows.append({
                "Товар": p,
                "Цена, ₽": round(float(np.mean(last_prices)), 2),
                "Цена (LightGBM), ₽": round(safe_mean_or_none(nl_prices) or 0.0, 2),
                "Надёжн.": "да" if safe_all_check(nl_rel_entity) else "нет",
                "Прогноз прибыли, ₽": round(pred_nl_sum, 2),
                "Δ к факту, %": round(growth_nl_pct, 1),
                "Диагностика": "; ".join(warn_unique) if warn_unique else "OK",
            })

    a_agg, b_agg, opt_agg, rel_agg = fit_regression_aggregate_daily(work_df)

    return {
        "rule_rows": rule_rows,
        "reg_rows": reg_rows,
        "nl_rows": nl_rows,
        "mean_rec_rules": safe_mean_or_none([x for x in rec_prices_rules if x is not None]) or 0.0,
        "mean_opt_reg": safe_mean_or_none([x for x in opts if x is not None]) or 0.0,
        "mean_opt_nl": safe_mean_or_none([x for x in nl_opts if x is not None]) or 0.0,
        "growth_rules_pct": safe_growth_pct(total_pred_rules, total_profit_actual),
        "growth_reg_pct": safe_growth_pct(total_pred_reg, total_profit_actual),
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
    product_df = prod_df[prod_df["product"] == selected_product].sort_values("date")
    entity_frames = _iter_entities(product_df if not product_df.empty else work_df)

    last_prices: list[float] = []
    last_comp_prices: list[float] = []
    last_profit_values: list[float] = []
    last_cogs_values: list[float] = []
    rule_prices: list[float] = []
    rule_names: list[str] = []
    reg_prices: list[float] = []
    reg_rel_flags: list[bool] = []
    reg_a_vals: list[float] = []
    reg_b_vals: list[float] = []
    nl_prices: list[float] = []
    nl_rel_flags: list[bool] = []
    nl_warnings: list[str] = []
    current_profit_sum = 0.0
    pred_rules_sum = 0.0
    pred_reg_sum = 0.0
    pred_nl_sum = 0.0

    for ent_df in entity_frames:
        valid_df = _exclude_oos_rows(ent_df)
        ent_used = valid_df if not valid_df.empty else ent_df
        if ent_used.empty:
            logger.debug(f"Пропущена сущность товара '{selected_product}' из-за отсутствия валидных данных")
            continue
        last_row_ent = ent_used.iloc[-1]
        prev_sales = ent_used["sales"].iloc[:-1]
        avg7 = prev_sales.tail(7).mean() if len(prev_sales) > 0 else last_row_ent["sales"]
        current_profit_ent = _stable_profit_baseline(ent_used)
        current_profit_sum += current_profit_ent

        last_prices.append(float(last_row_ent["our_price"]))
        last_comp_prices.append(float(last_row_ent.get("competitor_price", last_row_ent["our_price"])))
        last_profit_values.append(float(last_row_ent.get("profit", 0.0)))
        last_cogs_values.append(float(last_row_ent.get("cogs", 0.0)))

        rec_price_ent, rule_name_ent = apply_rules(last_row_ent, avg7)
        fc_rules_ent = forecast(selected_product, rec_price_ent, current_profit_ent)
        rule_prices.append(float(rec_price_ent))
        rule_names.append(rule_name_ent)
        pred_rules_sum += float(fc_rules_ent["forecast_profit"])

        a_ent, b_ent, opt_price_ent, rel_ent = fit_regression(ent_used, selected_product)
        fc_reg_ent = forecast(selected_product, opt_price_ent, current_profit_ent, regression_params=(a_ent, b_ent))
        reg_prices.append(float(opt_price_ent))
        reg_rel_flags.append(bool(rel_ent))
        reg_a_vals.append(float(a_ent))
        reg_b_vals.append(float(b_ent))
        pred_reg_sum += float(fc_reg_ent["forecast_profit"])

        cogs_ent = float(last_row_ent.get("cogs", 0.0))
        if include_lightgbm:
            nl_res_ent = recommend_price_lightgbm(
                history_df=ent_used,
                next_date=pd.Timestamp(last_row_ent["date"]) + pd.Timedelta(days=1),
                last_price=float(last_row_ent["our_price"]),
                competitor_price=float(last_row_ent.get("competitor_price", last_row_ent["our_price"])),
                cogs=cogs_ent,
            )
        else:
            nl_res_ent = {
                "recommended_price": float(last_row_ent["our_price"]),
                "reliable": False,
                "warnings": ["Расчёт LightGBM отключён."],
                "pred_sales": float(last_row_ent["sales"]),
            }
        pred_profit_nl_ent = float(nl_res_ent["pred_sales"]) * (float(nl_res_ent["recommended_price"]) - cogs_ent)
        pred_nl_sum += pred_profit_nl_ent
        nl_prices.append(float(nl_res_ent["recommended_price"]))
        nl_rel_flags.append(bool(nl_res_ent["reliable"]))
        nl_warnings.extend(list(nl_res_ent["warnings"]))

    if not last_prices:
        return {}

    rec_price_rules = safe_mean_or_none(rule_prices) or float(np.mean(last_prices))
    opt_price_reg = safe_mean_or_none(reg_prices) or float(np.mean(last_prices))
    opt_price_nl = safe_mean_or_none(nl_prices) or float(np.mean(last_prices))
    growth_rules = safe_growth_pct(pred_rules_sum, current_profit_sum)
    growth_reg = safe_growth_pct(pred_reg_sum, current_profit_sum)
    growth_nl = safe_growth_pct(pred_nl_sum, current_profit_sum)
    last_row = pd.Series({
        "our_price": float(np.mean(last_prices)),
        "competitor_price": float(np.mean(last_comp_prices)) if last_comp_prices else float(np.mean(last_prices)),
        "profit": float(np.sum(last_profit_values)),
        "cogs": float(np.mean(last_cogs_values)) if last_cogs_values else 0.0,
    })
    rule_name = format_rule_names(rule_names)
    unique_nl_warnings = list(dict.fromkeys([w for w in nl_warnings if w]))

    return {
        "last_row": last_row,
        "rec_price_rules": rec_price_rules,
        "rule_name": rule_name,
        "fc_rules": {
            "forecast_profit": round(pred_rules_sum, 2),
            "growth_pct": round(growth_rules, 1),
        },
        "a": safe_mean_or_none(reg_a_vals) or 0.0,
        "b": safe_mean_or_none(reg_b_vals) or 0.0,
        "opt_price_reg": opt_price_reg,
        "is_reg_reliable": safe_all_check(reg_rel_flags),
        "fc_reg": {
            "forecast_profit": round(pred_reg_sum, 2),
            "growth_pct": round(growth_reg, 1),
        },
        "opt_price_nl": opt_price_nl,
        "is_nl_reliable": safe_all_check(nl_rel_flags),
        "nl_warnings": unique_nl_warnings,
        "fc_nl": {
            "forecast_profit": round(pred_nl_sum, 2),
            "growth_pct": round(growth_nl, 1),
        },
    }
