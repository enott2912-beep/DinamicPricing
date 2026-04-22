"""
Генератор синтетических данных продаж.

Создаёт файл data/sales_history.csv с историей продаж за N_DAYS дней
для каждого товара из config.PRODUCTS.

Формула продаж:
    sales = base_sales - elasticity * (our_price - base_price) + noise
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Импорты из центральной модели
sys.path.append(str(Path(__file__).parent.parent))
from model.pricing import PRODUCTS, SEED
from model.math_engine import (
    calc_competitor_1_prices,
    calc_competitor_2_prices,
    calc_demand_rules,
)

STORES = [
    {"id": 1, "name": "СеверМаркет", "profile": "Премиум"},
    {"id": 2, "name": "ГородПлюс", "profile": "У дома"},
    {"id": 3, "name": "ЛентаНова", "profile": "Дискаунтер"},
    {"id": 4, "name": "ДомПокупок", "profile": "У дома"},
    {"id": 5, "name": "РитейлХаб", "profile": "Премиум"},
]

PRODUCT_BRANDS = {
    "Молоко": ["БелыйЛуг", "MilkWay", "СелоПрайм"],
    "Хлеб": ["Хлебница", "ПечкинДом", "ЗерноЛайн"],
    "Сок": ["FruitFlow", "СадовыйБерег", "VitaDrop"],
    "Кофе": ["RoastPoint", "BeanCraft", "MorningCup"],
    "Шоколад": ["Милка", "Орео", "АльпенГолд"],
}

STORE_PROFILE_FACTORS = {
    "Премиум": {"elasticity": 0.75, "base_sales": 0.88, "our_price_bias": 1.05},
    "Дискаунтер": {"elasticity": 1.35, "base_sales": 1.15, "our_price_bias": 0.95},
    "У дома": {"elasticity": 1.00, "base_sales": 1.00, "our_price_bias": 1.00},
}

SEASONALITY_PHASE = {
    "Кофе": -np.pi / 2,  # зимний пик
    "Сок": np.pi / 2,    # летний пик
    "Молоко": 0.0,
    "Хлеб": 0.0,
    "Шоколад": -np.pi / 3,
}

OOS_PROBABILITY = 0.05


def generate_all_data(n_days: int, start_date: datetime) -> pd.DataFrame:
    """Векторизованная генерация данных по иерархии Магазин -> Бренд -> Товар."""
    rng = np.random.default_rng(SEED)

    product_names = list(PRODUCTS.keys())
    n_products = len(product_names)
    n_stores = len(STORES)
    base_prices = np.array([PRODUCTS[p]['base_price'] for p in product_names])
    base_sales = np.array([PRODUCTS[p]['base_sales'] for p in product_names])
    elasticities = np.array([PRODUCTS[p]['elasticity'] for p in product_names])
    product_ids = np.array([PRODUCTS[p]['id'] for p in product_names])
    cogs = np.array([PRODUCTS[p].get('cogs', 0.0) for p in product_names])

    product_cfg = {
        name: {"id": pid, "base_price": bp, "base_sales": bs, "elasticity": el, "cogs": cg}
        for name, pid, bp, bs, el, cg in zip(product_names, product_ids, base_prices, base_sales, elasticities, cogs)
    }
    sku_counter = 1000
    entity_rows = []
    for store in STORES:
        profile_name = store["profile"]
        profile = STORE_PROFILE_FACTORS[profile_name]
        for product_name in product_names:
            brands = PRODUCT_BRANDS.get(product_name, [])
            for brand_idx, brand_name in enumerate(brands, start=1):
                cfg = product_cfg[product_name]
                # Бренды одного товара отличаются по спросу и чувствительности к цене.
                brand_sales_factor = 0.86 + 0.12 * brand_idx + rng.uniform(-0.03, 0.03)
                brand_elasticity_factor = 0.90 + 0.06 * brand_idx + rng.uniform(-0.02, 0.02)
                adjusted_base_sales = max(5.0, cfg["base_sales"] * profile["base_sales"] * brand_sales_factor)
                adjusted_elasticity = max(0.2, cfg["elasticity"] * profile["elasticity"] * brand_elasticity_factor)
                sku_counter += 1
                entity_rows.append({
                    "store_id": store["id"],
                    "store": store["name"],
                    "store_profile": profile_name,
                    "brand_id": sku_counter,
                    "brand": brand_name,
                    "product_id": sku_counter,
                    "product": product_name,
                    "base_price": cfg["base_price"],
                    "base_sales": adjusted_base_sales,
                    "elasticity": adjusted_elasticity,
                    "our_price_bias": profile["our_price_bias"],
                    "cogs": cfg["cogs"],
                })

    entity_df = pd.DataFrame(entity_rows)
    n_entities = len(entity_df)

    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')

    base_prices_entities = entity_df["base_price"].to_numpy(dtype=float)
    base_sales_entities = entity_df["base_sales"].to_numpy(dtype=float)
    elasticities_entities = entity_df["elasticity"].to_numpy(dtype=float)

    # Цены
    base_price_mat = np.tile(base_prices_entities, (n_days, 1))
    our_price_bias = entity_df["our_price_bias"].to_numpy(dtype=float)
    our_prices = np.round(
        base_price_mat
        * np.tile(our_price_bias, (n_days, 1))
        * rng.uniform(0.92, 1.08, size=(n_days, n_entities)),
        2,
    )

    comp1_base = base_prices_entities * 1.03
    comp2_base = base_prices_entities * 0.93
    competitor_1_prices = np.zeros((n_days, n_entities))
    competitor_2_prices = np.zeros((n_days, n_entities))
    competitor_1_prices[0] = np.maximum(
        1,
        comp1_base + rng.normal(0, 0.02 * base_prices_entities, size=n_entities)
    )
    competitor_2_prices[0] = np.maximum(
        1,
        comp2_base + rng.normal(0, 0.03 * base_prices_entities, size=n_entities)
    )
    aggressive_mask = rng.random(n_entities) < 0.45

    for i in range(1, n_days):
        noise_1 = rng.normal(0, 0.015 * base_prices_entities, size=n_entities)
        noise_2 = rng.normal(0, 0.020 * base_prices_entities, size=n_entities)
        chaos_step = aggressive_mask & (rng.random(n_entities) < 0.15)
        competitor_1_prices[i] = calc_competitor_1_prices(
            competitor_1_prices[i - 1], comp1_base, our_prices[i - 1], noise_1
        )
        competitor_2_prices[i] = calc_competitor_2_prices(
            competitor_2_prices[i - 1], comp2_base, base_prices_entities * 0.90, noise_2, chaos_step
        )
    competitor_prices = np.round((competitor_1_prices + competitor_2_prices) / 2.0, 2)

    noise_scale = np.maximum(2.0, 0.08 * base_sales_entities)
    noise_mat = np.tile(noise_scale, (n_days, 1))
    noise = rng.normal(0, noise_mat, size=(n_days, n_entities))

    elast_mat = np.tile(elasticities_entities, (n_days, 1))
    base_sales_mat = np.tile(base_sales_entities, (n_days, 1))

    day_of_year = dates.dayofyear.to_numpy(dtype=float)
    omega = 2 * np.pi / 365.0
    phase = np.array([SEASONALITY_PHASE.get(p, 0.0) for p in entity_df["product"].to_numpy()], dtype=float)
    seasonal_multiplier = 1.0 + 0.16 * np.sin(day_of_year[:, None] * omega + phase[None, :])
    seasonally_adjusted_base_sales = np.maximum(0.0, base_sales_mat * seasonal_multiplier)

    sales = calc_demand_rules(
        our_prices=our_prices,
        competitor_prices=np.minimum(competitor_1_prices, competitor_2_prices),
        base_prices=base_price_mat,
        base_sales=seasonally_adjusted_base_sales,
        elasticities=elast_mat,
        noise=noise
    )
    oos_mask = rng.random((n_days, n_entities)) < OOS_PROBABILITY
    sales[oos_mask] = 0.0

    revenue = np.round(sales * our_prices, 2)
    cogs_entity = entity_df["cogs"].to_numpy(dtype=float)
    cogs_mat = np.tile(cogs_entity, (n_days, 1))
    profit = np.round(revenue - sales * cogs_mat, 2)

    date_col = np.repeat(dates, n_entities)
    store_id_col = np.tile(entity_df["store_id"].to_numpy(), n_days)
    store_col = np.tile(entity_df["store"].to_numpy(), n_days)
    brand_id_col = np.tile(entity_df["brand_id"].to_numpy(), n_days)
    brand_col = np.tile(entity_df["brand"].to_numpy(), n_days)
    product_id_col = np.tile(entity_df["product_id"].to_numpy(), n_days)
    product_col = np.tile(entity_df["product"].to_numpy(), n_days)
    cogs_col = np.tile(cogs_entity, n_days)

    df = pd.DataFrame({
        'date': date_col,
        'store_id': store_id_col,
        'store': store_col,
        'store_profile': np.tile(entity_df["store_profile"].to_numpy(), n_days),
        'brand_id': brand_id_col,
        'brand': brand_col,
        'product_id': product_id_col,
        'product': product_col,
        'our_price': our_prices.flatten(),
        'competitor_1_price': competitor_1_prices.flatten(),
        'competitor_2_price': competitor_2_prices.flatten(),
        'competitor_price': competitor_prices.flatten(),
        'is_oos': oos_mask.flatten(),
        'sales': sales.flatten(),
        'revenue': revenue.flatten(),
        'cogs': cogs_col,
        'profit': profit.flatten()
    })

    return df.sort_values(['date', 'store', 'brand', 'product']).reset_index(drop=True)


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет DataFrame в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Данные сохранены: {path.absolute()}")


def main(n_days: int = 100) -> None:
    """Точка входа."""
    np.random.seed(SEED)
    n_days = int(max(1, n_days))
    
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'sales_history.csv'
    predict_path = base_dir / 'data' / 'predict_sales.csv'

    # Каждый запуск генератора полностью перезаписывает историю.
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days - 1)
    print(f"Полная перегенерация истории за {n_days} дней...")
    df = generate_all_data(n_days, start_date)

    save_data(df, data_path)
    # Прогнозы должны храниться отдельно и очищаться при новой генерации истории.
    pd.DataFrame(columns=df.columns).to_csv(predict_path, index=False)
    
    print("\nПример данных:")
    print(df.head())
    print("\nКорреляция Цена-Продажи (должна быть < 0):")
    for p in PRODUCTS:
        sub = df[df['product'] == p]
        corr = sub['our_price'].corr(sub['sales'])
        print(f" - {p}: {corr:.4f}")

    print("\nГенерация завершена успешно.")


if __name__ == '__main__':
    main()
