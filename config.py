# config.py
# Конфигурационный файл проекта с общими настройками и константами

# Параметры генерации данных
SEED = 42
N_DAYS = 100

# Параметры товаров (id, базовая цена, эластичность, базовые продажи при base_price)
PRODUCTS = {
    'Молоко': {'id': 1, 'base_price': 80, 'elasticity': 2.0, 'base_sales': 300},
    'Хлеб': {'id': 2, 'base_price': 50, 'elasticity': 1.5, 'base_sales': 250},
    'Сок': {'id': 3, 'base_price': 120, 'elasticity': 3.0, 'base_sales': 150},
    'Кофе': {'id': 4, 'base_price': 450, 'elasticity': 1.2, 'base_sales': 80},
    'Шоколад': {'id': 5, 'base_price': 100, 'elasticity': 2.5, 'base_sales': 200}
}
