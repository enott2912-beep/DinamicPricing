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
sys.path.append(str(Path(__file__).parent.parent))  # noqa: E402
from model.pricing import PRODUCTS, SEED  # noqa: E402
from model.math_engine import (  # noqa: E402
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

SEGMENT_LEN_MIN = 7
SEGMENT_LEN_MAX = 21
SEGMENT_AMPLITUDE = {
    "Кофе": 0.25,
    "Сок": 0.18,
    "Шоколад": 0.15,
    "Молоко": 0.08,
    "Хлеб": 0.08,
}
DEFAULT_AMPLITUDE = 0.08


def _build_entity_df(rng: np.random.Generator) -> pd.DataFrame:
    product_names = list(PRODUCTS.keys())
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
    return pd.DataFrame(entity_rows)


def _finalize_dataframe(
    dates: pd.DatetimeIndex,
    entity_df: pd.DataFrame,
    our_prices: np.ndarray,
    competitor_1_prices: np.ndarray,
    competitor_2_prices: np.ndarray,
    competitor_prices: np.ndarray,
    oos_mask: np.ndarray,
    sales: np.ndarray,
) -> pd.DataFrame:
    n_days = len(dates)
    n_entities = len(entity_df)
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


def _simulate_prices_segmented(
    entity_df: pd.DataFrame,
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Генерирует автокоррелированные сегменты цен вместо независимого дневного шума."""
    n_entities = len(entity_df)
    base_prices = entity_df["base_price"].to_numpy(dtype=float)
    price_bias = entity_df["our_price_bias"].to_numpy(dtype=float)
    products = entity_df["product"].to_numpy()
    our_prices = np.zeros((n_days, n_entities))

    for entity_idx in range(n_entities):
        amplitude = SEGMENT_AMPLITUDE.get(str(products[entity_idx]), DEFAULT_AMPLITUDE)
        day = 0
        while day < n_days:
            segment_len = int(rng.integers(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX + 1))
            segment_len = min(segment_len, n_days - day)
            price_shift = rng.uniform(-amplitude, amplitude)
            segment_base = base_prices[entity_idx] * price_bias[entity_idx] * (1.0 + price_shift)
            intra_segment_noise = rng.uniform(0.985, 1.015, size=segment_len)
            our_prices[day:day + segment_len, entity_idx] = np.round(segment_base * intra_segment_noise, 2)
            day += segment_len

    return our_prices


def _simulate_prices(
    entity_df: pd.DataFrame,
    n_days: int,
    rng: np.random.Generator,
    use_legacy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_entities = len(entity_df)
    base_prices_entities = entity_df["base_price"].to_numpy(dtype=float)
    if use_legacy:
        base_price_mat = np.tile(base_prices_entities, (n_days, 1))
        our_price_bias = entity_df["our_price_bias"].to_numpy(dtype=float)
        our_prices = np.round(
            base_price_mat
            * np.tile(our_price_bias, (n_days, 1))
            * rng.uniform(0.92, 1.08, size=(n_days, n_entities)),
            2,
        )
    else:
        our_prices = _simulate_prices_segmented(entity_df, n_days, rng)

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
    return our_prices, competitor_1_prices, competitor_2_prices, competitor_prices


def generate_all_data(n_days: int, start_date: datetime) -> pd.DataFrame:
    """Векторизованная генерация данных по иерархии Магазин -> Бренд -> Товар."""
    rng = np.random.default_rng(SEED)

    entity_df = _build_entity_df(rng)
    n_entities = len(entity_df)

    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')

    base_sales_entities = entity_df["base_sales"].to_numpy(dtype=float)
    elasticities_entities = entity_df["elasticity"].to_numpy(dtype=float)
    base_price_mat = np.tile(entity_df["base_price"].to_numpy(dtype=float), (n_days, 1))
    our_prices, competitor_1_prices, competitor_2_prices, competitor_prices = _simulate_prices(entity_df, n_days, rng)

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

    return _finalize_dataframe(
        dates,
        entity_df,
        our_prices,
        competitor_1_prices,
        competitor_2_prices,
        competitor_prices,
        oos_mask,
        sales,
    )


def generate_all_data_nonlinear(n_days: int, start_date: datetime) -> pd.DataFrame:
    """Нелинейный генератор для экспериментального режима LightGBM."""
    rng = np.random.default_rng(SEED)
    entity_df = _build_entity_df(rng)
    n_entities = len(entity_df)

    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    base_prices_entities = entity_df["base_price"].to_numpy(dtype=float)
    base_sales_entities = entity_df["base_sales"].to_numpy(dtype=float)
    elasticities_entities = entity_df["elasticity"].to_numpy(dtype=float)

    base_price_mat = np.tile(base_prices_entities, (n_days, 1))
    our_prices, competitor_1_prices, competitor_2_prices, competitor_prices = _simulate_prices(entity_df, n_days, rng)
    best_comp = np.minimum(competitor_1_prices, competitor_2_prices)

    day_of_year = dates.dayofyear.to_numpy(dtype=float)
    omega = 2 * np.pi / 365.0
    phase = np.array([SEASONALITY_PHASE.get(p, 0.0) for p in entity_df["product"].to_numpy()], dtype=float)
    seasonal = 1.0 + 0.16 * np.sin(day_of_year[:, None] * omega + phase[None, :])

    rel_price = our_prices / np.maximum(base_price_mat, 1.0)
    comp_ratio = our_prices / np.maximum(best_comp, 1.0)
    # Нелинейные эффекты: насыщение по цене + усиленная реакция на сильный undercut.
    sat_effect = np.exp(-2.8 * np.maximum(rel_price - 1.0, 0.0))
    undercut_penalty = np.where(comp_ratio > 1.08, 1.0 - 0.65 * (comp_ratio - 1.08) ** 2, 1.0)
    undercut_penalty = np.clip(undercut_penalty, 0.45, 1.15)
    # Взаимодействие цена x сезон.
    season_price_inter = 1.0 - 0.18 * np.maximum(rel_price - 1.0, 0.0) * np.maximum(seasonal - 1.0, 0.0)
    season_price_inter = np.clip(season_price_inter, 0.6, 1.2)

    linear_core = (
        base_sales_entities[None, :]
        - elasticities_entities[None, :] * (our_prices - base_price_mat)
        - 0.5 * elasticities_entities[None, :] * (our_prices - best_comp)
    )
    nonlinear_sales = np.maximum(0.0, linear_core) * seasonal * sat_effect * undercut_penalty * season_price_inter
    noise_scale = np.maximum(2.0, 0.09 * base_sales_entities)
    noise = rng.normal(0, np.tile(noise_scale, (n_days, 1)), size=(n_days, n_entities))
    sales = np.maximum(0.0, np.round(nonlinear_sales + noise))

    oos_mask = rng.random((n_days, n_entities)) < OOS_PROBABILITY
    sales[oos_mask] = 0.0
    competitor_prices = np.round(competitor_prices, 2)

    return _finalize_dataframe(
        dates,
        entity_df,
        our_prices,
        competitor_1_prices,
        competitor_2_prices,
        competitor_prices,
        oos_mask,
        sales,
    )


def generate_elasticity_validation_data(
    product: str = "Сок",
    store: str = "СеверМаркет",
    n_steps: int = 6,
    days_per_step: int = 14,
    seed: int = 0,
) -> pd.DataFrame:
    """Контролируемые ступенчатые изменения цены для проверки эластичности."""
    if product not in PRODUCTS:
        raise ValueError(f"Неизвестный продукт: {product}. Допустимы: {list(PRODUCTS)}")

    product_cfg = PRODUCTS[product]
    base_price = product_cfg["base_price"]
    base_sales = product_cfg["base_sales"]
    elasticity = product_cfg["elasticity"]
    true_b = elasticity * base_sales / base_price

    rng = np.random.default_rng(seed)
    shifts = np.linspace(-0.25, 0.25, n_steps)
    records = []

    for step_idx, shift in enumerate(shifts):
        price = base_price * (1.0 + shift)
        for day in range(days_per_step):
            noise = rng.normal(0, base_sales * 0.03)
            sales = max(0.0, base_sales - true_b * (price - base_price) + noise)
            records.append({
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=step_idx * days_per_step + day),
                "store": store,
                "product": product,
                "our_price": round(float(price), 2),
                "competitor_price": round(float(base_price), 2),
                "sales": round(float(sales), 1),
                "cogs": round(float(base_price * 0.6), 2),
                "is_oos": False,
                "true_B": round(float(true_b), 4),
            })

    return pd.DataFrame(records)


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет DataFrame в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Данные сохранены: {path.absolute()}")


def main(n_days: int = 100, mode: str = "basic") -> None:
    """Точка входа."""
    np.random.seed(SEED)
    n_days = int(max(1, n_days))

    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'sales_history.csv'
    predict_path = base_dir / 'data' / 'predict_sales.csv'
    mode_path = base_dir / 'data' / 'generation_mode.txt'

    # Каждый запуск генератора полностью перезаписывает историю.
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days - 1)
    mode = (mode or "basic").strip().lower()
    print(f"Полная перегенерация истории за {n_days} дней... (режим: {mode})")
    if mode == "experimental":
        df = generate_all_data_nonlinear(n_days, start_date)
    else:
        df = generate_all_data(n_days, start_date)

    save_data(df, data_path)
    mode_path.parent.mkdir(parents=True, exist_ok=True)
    mode_path.write_text("experimental" if mode == "experimental" else "baseline", encoding="utf-8")
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
