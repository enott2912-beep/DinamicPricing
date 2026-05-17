"""
Генерация тестового CSV с произвольными SKU (не из PRODUCTS) для проверки загрузки.
Запуск: python scripts/generate_custom_csv_sample.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.math_engine import calc_demand_rules

SEED = 42
N_DAYS = 100
N_PRODUCTS = 50
N_STORES = 3
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "custom_products_test_15000.csv"

STORES = [
    (101, "Северный", "У дома"),
    (102, "Центральный", "Супермаркет"),
    (103, "Южный", "Дискаунтер"),
]

PRODUCT_NAMES = [
    "Яблоки Гала",
    "Груши Конференция",
    "Бананы",
    "Апельсины",
    "Мандарины",
    "Картофель",
    "Морковь",
    "Лук репчатый",
    "Помидоры",
    "Огурцы",
    "Капуста",
    "Рис длиннозёрный",
    "Гречка",
    "Макароны спагетти",
    "Овсяные хлопья",
    "Мука пшеничная",
    "Сахар",
    "Соль",
    "Подсолнечное масло",
    "Оливковое масло",
    "Курица охлаждённая",
    "Говядина тушёная",
    "Свинина вырезка",
    "Филе индейки",
    "Лосось замороженный",
    "Сельдь",
    "Сыр Российский",
    "Творог 5%",
    "Йогурт натуральный",
    "Сметана 20%",
    "Кефир 2.5%",
    "Яйца С0",
    "Масло сливочное",
    "Колбаса варёная",
    "Сосиски",
    "Пельмени",
    "Кетчуп",
    "Майонез",
    "Горчица",
    "Чай чёрный",
    "Какао",
    "Печенье овсяное",
    "Вафли",
    "Орехи микс",
    "Вода минеральная",
    "Сок яблочный",
    "Лимонад",
    "Квас",
    "Пиво светлое",
    "Вино красное",
]


def _product_catalog(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(PRODUCT_NAMES[:N_PRODUCTS], start=1):
        base_price = float(rng.uniform(35, 520))
        base_sales = float(rng.uniform(40, 420))
        elasticity = float(rng.uniform(0.8, 3.5))
        cogs = round(base_price * rng.uniform(0.55, 0.82), 2)
        rows.append(
            {
                "product_id": 10_000 + i,
                "product": name,
                "brand_id": 200 + (i % 8),
                "brand": f"Бренд_{(i % 8) + 1}",
                "base_price": base_price,
                "base_sales": base_sales,
                "elasticity": elasticity,
                "cogs": cogs,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    catalog = _product_catalog(rng)
    entities = []
    eid = 0
    for _, prod in catalog.iterrows():
        for store_id, store, profile in STORES:
            eid += 1
            entities.append(
                {
                    "entity_id": eid,
                    "store_id": store_id,
                    "store": store,
                    "store_profile": profile,
                    **prod.to_dict(),
                }
            )
    entity_df = pd.DataFrame(entities)
    n_entities = len(entity_df)
    n_days = N_DAYS
    expected_rows = n_entities * n_days
    if expected_rows != 15_000:
        raise RuntimeError(f"Ожидалось 15000 строк, получится {expected_rows}")

    end = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(end - timedelta(days=n_days - 1), periods=n_days, freq="D")

    base_prices = entity_df["base_price"].to_numpy(dtype=float)
    base_sales = entity_df["base_sales"].to_numpy(dtype=float)
    elasticities = entity_df["elasticity"].to_numpy(dtype=float)
    cogs = entity_df["cogs"].to_numpy(dtype=float)

    base_price_mat = np.tile(base_prices, (n_days, 1))
    drift = rng.normal(0, 0.012, size=(n_days, n_entities))
    our_prices = np.maximum(1.0, base_price_mat * (1.0 + np.cumsum(drift, axis=0)))
    our_prices = np.round(our_prices, 2)

    comp_noise = rng.normal(0, 0.015, size=(n_days, n_entities))
    competitor_1 = np.round(our_prices * (1.02 + comp_noise), 2)
    competitor_2 = np.round(our_prices * (0.97 + comp_noise * 0.8), 2)
    competitor_price = np.minimum(competitor_1, competitor_2)

    noise = rng.normal(0, np.maximum(2.0, 0.08 * base_sales), size=(n_days, n_entities))
    sales = calc_demand_rules(
        our_prices,
        competitor_price,
        base_price_mat,
        np.tile(base_sales, (n_days, 1)),
        np.tile(elasticities, (n_days, 1)),
        noise,
    )
    oos = rng.random((n_days, n_entities)) < 0.04
    sales = sales.astype(float)
    sales[oos] = 0.0

    revenue = np.round(sales * our_prices, 2)
    profit = np.round(revenue - sales * cogs, 2)

    rep = lambda col: np.tile(entity_df[col].to_numpy(), n_days)
    out = pd.DataFrame(
        {
            "date": np.repeat(dates.values, n_entities),
            "store_id": rep("store_id"),
            "store": rep("store"),
            "store_profile": rep("store_profile"),
            "brand_id": rep("brand_id"),
            "brand": rep("brand"),
            "product_id": rep("product_id"),
            "product": rep("product"),
            "our_price": our_prices.reshape(-1),
            "competitor_1_price": competitor_1.reshape(-1),
            "competitor_2_price": competitor_2.reshape(-1),
            "competitor_price": competitor_price.reshape(-1),
            "is_oos": oos.reshape(-1),
            "sales": sales.reshape(-1),
            "revenue": revenue.reshape(-1),
            "cogs": np.tile(cogs, n_days),
            "profit": profit.reshape(-1),
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"Сохранено: {OUTPUT}")
    print(f"Строк: {len(out)}, SKU: {out['product'].nunique()}, дней: {out['date'].nunique()}")


if __name__ == "__main__":
    main()
