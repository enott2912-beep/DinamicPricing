# Сводная таблица unit-тестов (P0 + P1 + LightGBM)

Запуск из каталога `DinamicPricing/`:

```bash
pip install -r requirements-dev.txt
pytest
```

Конфигурация: `pytest.ini`. Кэш pytest (`.pytest_cache/`) в git не коммитится.

## Структура каталога `tests/`

| Файл | Модуль под тестом | Тестов |
|------|-------------------|--------|
| `conftest.py` | — (фикстуры) | 0 |
| `test_math_engine.py` | `model/math_engine.py` | 6 |
| `test_rules_validator.py` | `model/rules_validator.py` | 8 |
| `test_rules_engine.py` | `model/rules_engine.py` | 6 |
| `test_pricing_regression.py` | `model/pricing.py` (линейная регрессия) | 8 |
| `test_pricing_forecast.py` | `model/pricing.py` (прогноз, правила) | 3 |
| `test_data_validation.py` | `ui/data_manager.py` → `_validate_loaded_data` | 5 |
| `test_pricing_lightgbm.py` | `model/pricing.py` (LightGBM) | 9 |
| `test_simulate.py` | `model/pricing.py` → `simulate` | 16 |
| **Итого** | | **61** |

### `conftest.py` (не тесты)

| Фикстура | Назначение |
|----------|------------|
| `minimal_sales_df` | 8 дней истории по SKU «Молоко» с согласованными revenue/profit |
| `lgbm_training_df` | 70 дней с вариативными ценами — достаточно для обучения LightGBM |
| `sim_history_df` | 21 день, одна сущность с иерархией store/brand — для `simulate` |
| `sim_history_multi_entity` | 14 дней, «Молоко» + «Кофе» — фильтр SKU и число строк |
| `sample_rules` | Два правила для `RuleEngine` (первое всегда срабатывает) |
| `rule_engine` | `RuleEngine` с `sample_rules` во временной директории |

---

## Полная таблица тестов

| № | Файл | Тест | Тестируемая функция | Что проверяет |
|---|------|------|---------------------|---------------|
| 1 | `test_math_engine.py` | `test_calc_competitor_1_prices_minimum_one` | `calc_competitor_1_prices` | Все цены конкурента 1 после пересчёта не ниже 1 ₽. |
| 2 | `test_math_engine.py` | `test_calc_competitor_2_prices_minimum_one` | `calc_competitor_2_prices` | То же для конкурента 2 (обычный и «хаотичный» режим). |
| 3 | `test_math_engine.py` | `test_calc_demand_rules_non_negative` | `calc_demand_rules` | Спрос по эвристике не отрицательный. |
| 4 | `test_math_engine.py` | `test_calc_demand_regression_matches_input_length` | `calc_demand_regression` | Размер вектора спроса совпадает с числом цен на входе. |
| 5 | `test_math_engine.py` | `test_calc_demand_rules_higher_price_lower_demand` | `calc_demand_rules` | При росте нашей цены (без шума) спрос падает. |
| 6 | `test_math_engine.py` | `test_calc_competitor_1_increases_when_our_price_increases` | `calc_competitor_1_prices` | При более высокой нашей цене цена follower-конкурента не ниже, чем при низкой. |
| 7 | `test_rules_validator.py` | `test_validate_rule_valid_minimal` | `validate_rule` | Корректное правило (name, condition, action) принимается. |
| 8 | `test_rules_validator.py` | `test_validate_rule_missing_name` | `validate_rule` | Отсутствие названия → отклонение. |
| 9 | `test_rules_validator.py` | `test_validate_rule_missing_condition` | `validate_rule` | Отсутствие условия → отклонение. |
| 10 | `test_rules_validator.py` | `test_validate_rule_missing_action` | `validate_rule` | Отсутствие действия → отклонение. |
| 11 | `test_rules_validator.py` | `test_validate_rule_unknown_variable` | `validate_rule` | Неизвестная переменная в условии → отклонение. |
| 12 | `test_rules_validator.py` | `test_validate_rule_negative_action_price` | `validate_rule` | Действие даёт отрицательную цену → отклонение. |
| 13 | `test_rules_validator.py` | `test_validate_rules_list_too_many` | `validate_rules_list` | Больше 50 правил в списке → невалидно. |
| 14 | `test_rules_validator.py` | `test_validate_rules_list_collects_multiple_errors` | `validate_rules_list` | Несколько битых правил → несколько сообщений об ошибках. |
| 15 | `test_rules_engine.py` | `test_rule_engine_evaluate_first_matching_rule` | `RuleEngine.evaluate` | Срабатывает первое подходящее правило; цена пересчитана. |
| 16 | `test_rules_engine.py` | `test_rule_engine_evaluate_hold_when_no_match` | `RuleEngine.evaluate` | Ни одно условие не выполнено → `hold`, цена без изменений. |
| 17 | `test_rules_engine.py` | `test_rule_engine_evaluate_no_rules` | `RuleEngine.evaluate` | Пустой список правил → `no_rules`. |
| 18 | `test_rules_engine.py` | `test_rule_engine_margin_available_in_condition` | `RuleEngine.evaluate` | В условии доступна вычисленная `margin`; правило срабатывает. |
| 19 | `test_rules_engine.py` | `test_rule_engine_skips_broken_rule` | `RuleEngine.evaluate` | Ошибка в первом правиле → пропуск, срабатывает следующее. |
| 20 | `test_rules_engine.py` | `test_rule_engine_save_rules_rejects_invalid` | `RuleEngine.save_rules` | Невалидный список правил → `ValueError` при сохранении. |
| 21 | `test_pricing_regression.py` | `test_fit_linear_sales_vs_price_insufficient_data` | `_fit_linear_sales_vs_price` | Меньше 3 точек → модель ненадёжна, fallback-цена. |
| 22 | `test_pricing_regression.py` | `test_fit_linear_sales_vs_price_negative_slope_reliable` | `_fit_linear_sales_vs_price` | Sales падают с ценой → B>0, `reliable=True`, оптимум ≥ 1. |
| 23 | `test_pricing_regression.py` | `test_fit_linear_optimal_price_near_formula` | `_fit_linear_sales_vs_price` | Оптимальная цена близка к (A+B·COGS)/(2B) с учётом клиппинга. |
| 24 | `test_pricing_regression.py` | `test_fit_regression_filters_oos` | `fit_regression` | Строки `is_oos=True` с аномальными sales не портят обучение. |
| 25 | `test_pricing_regression.py` | `test_fit_regression_filters_zero_sales` | `fit_regression` | День с `sales=0` исключается из обучения. |
| 26 | `test_pricing_regression.py` | `test_predict_sales_regression_formula` | `predict_sales_regression` | Спрогноз = max(0, round(A−B·Price)) без шума. |
| 27 | `test_pricing_regression.py` | `test_predict_sales_regression_with_noise` | `predict_sales_regression` | То же с добавлением шума в формулу. |
| 28 | `test_pricing_regression.py` | `test_products_in_dataframe_order` | `products_in_dataframe` | Порядок SKU как в `PRODUCTS`, только присутствующие в df. |
| 29 | `test_pricing_forecast.py` | `test_forecast_with_regression_params` | `forecast` | С коэффициентами (A,B): sales, profit и growth_pct согласованы. |
| 30 | `test_pricing_forecast.py` | `test_forecast_without_regression_uses_products` | `forecast` | Без регрессии спрос считается из констант `PRODUCTS`. |
| 31 | `test_pricing_forecast.py` | `test_apply_rules_delegates_to_engine` | `apply_rules` | Контекст из строки DataFrame передаётся в движок правил. |
| 32 | `test_data_validation.py` | `test_validate_loaded_data_valid_minimal` | `_validate_loaded_data` | Минимально корректная строка CSV → DataFrame, без ошибки UI. |
| 33 | `test_data_validation.py` | `test_validate_loaded_data_missing_column` | `_validate_loaded_data` | Нет обязательной колонки → `None`, вызов `st.error`. |
| 34 | `test_data_validation.py` | `test_validate_loaded_data_revenue_mismatch` | `_validate_loaded_data` | `revenue` ≠ `sales × our_price` → отклонение. |
| 35 | `test_data_validation.py` | `test_validate_loaded_data_negative_price` | `_validate_loaded_data` | Отрицательная `our_price` → отклонение. |
| 36 | `test_data_validation.py` | `test_validate_loaded_data_bad_date` | `_validate_loaded_data` | Неразбираемая дата → отклонение. |
| 37 | `test_pricing_lightgbm.py` | `test_lgbm_data_warnings_short_history` | `_lgbm_data_warnings` | Короткая история → предупреждение о малом числе наблюдений. |
| 38 | `test_pricing_lightgbm.py` | `test_build_lgbm_training_frame_filters_oos` | `_build_lgbm_training_frame` | OOS и нулевые sales не попадают в обучающую выборку. |
| 39 | `test_pricing_lightgbm.py` | `test_fit_lightgbm_short_history_unreliable` | `fit_lightgbm_sales_model` | 8 дней → модель не обучается, `reliable=False`. |
| 40 | `test_pricing_lightgbm.py` | `test_fit_lightgbm_empty_after_clean` | `fit_lightgbm_sales_model` | Пустой df после очистки → модель `None`. |
| 41 | `test_pricing_lightgbm.py` | `test_fit_lightgbm_reliable_on_rich_history` | `fit_lightgbm_sales_model` | 70 дней с вариацией цен → модель обучена, `reliable=True`. |
| 42 | `test_pricing_lightgbm.py` | `test_recommend_price_lightgbm_with_pack_respects_step_limit` | `recommend_price_lightgbm_with_pack` | Рекомендованная цена в пределах дневного лимита ±2%. |
| 43 | `test_pricing_lightgbm.py` | `test_recommend_price_lightgbm_fallback_when_unreliable` | `recommend_price_lightgbm` | Слабые данные → цена не меняется (hold последней). |
| 44 | `test_pricing_lightgbm.py` | `test_predict_sales_lightgbm_non_negative` | `predict_sales_lightgbm_with_pack` | Прогноз спроса ≥ 0 при обученной модели. |
| 45 | `test_pricing_lightgbm.py` | `test_predict_sales_fallback_when_no_model` | `predict_sales_lightgbm_with_pack` | Без модели → среднее sales за 7 дней. |

| 46 | `test_simulate.py` | `test_simulate_appends_n_steps_days` | `simulate` | После `n_steps` уникальных дат стало ровно на столько больше. |
| 47 | `test_simulate.py` | `test_simulate_future_dates_after_history_max` | `simulate` | Все новые даты строго позже конца истории. |
| 48 | `test_simulate.py` | `test_simulate_output_schema` | `simulate` | В будущих строках есть цены, спрос, выручка, прибыль, конкуренты. |
| 49 | `test_simulate.py` | `test_simulate_invariants` | `simulate` | `our_price ≥ 1`, `sales ≥ 0`, `revenue ≥ 0` на прогнозных днях. |
| 50 | `test_simulate.py` | `test_simulate_accounting` | `simulate` | `revenue ≈ sales×price`, `profit ≈ revenue − sales×cogs`. |
| 51 | `test_simulate.py` | `test_simulate_reproducible_rules` | `simulate` | Два прогона `method=rules` с одним df → идентичный будущий хвост (SEED). |
| 52 | `test_simulate.py` | `test_simulate_empty_or_no_products` | `simulate` | Пустой df или неизвестный SKU → без новых дней / без падения. |
| 53 | `test_simulate.py` | `test_simulate_target_product_filter` | `simulate` | `target_product` ограничивает прогноз одним SKU. |
| 54 | `test_simulate.py` | `test_simulate_row_count` | `simulate` | Число прогнозных строк = `n_steps ×` число сущностей. |
| 55 | `test_simulate.py` | `test_simulate_regression_completes` | `simulate` | `method=regression` завершается на 21 дне истории. |
| 56 | `test_simulate.py` | `test_simulate_regression_daily_price_step_limit` | `simulate` | Шаг цены между днями в пределах `max_daily_price_change_pct`. |
| 57 | `test_simulate.py` | `test_simulate_regression_train_window_clipped` | `simulate` | Окно 999 дней при короткой истории не роняет симуляцию. |
| 58 | `test_simulate.py` | `test_simulate_rules_completes` | `simulate` | `method=rules` smoke на 5 шагов. |
| 59 | `test_simulate.py` | `test_simulate_rules_with_patched_engine` | `simulate` | С моком правила `price×1.1` первая будущая цена соответствует правилу. |
| 60 | `test_simulate.py` | `test_simulate_lightgbm_completes` | `simulate` | `method=lightgbm` на длинной истории — без падения, sales ≥ 0. |
| 61 | `test_simulate.py` | `test_simulate_lightgbm_short_history` | `simulate` | Короткая история + lightgbm — fallback, без падения. |

**Итого: 61 тест** (P0: 20, P1: 16, LightGBM: 9, simulate: 16). Если `lightgbm` не установлен, 11 тестов пропускаются (`pytest.skip`).

---

## Группировка по приоритету

| Приоритет | Файлы | Фокус |
|-----------|--------|--------|
| **P0** | `test_math_engine`, `test_rules_validator`, `test_rules_engine` | Формулы спроса и конкурентов; валидация и выполнение Rule Engine |
| **P1** | `test_pricing_regression`, `test_pricing_forecast`, `test_data_validation` | Линейная регрессия и прогноз; валидация загружаемого CSV |
| **LightGBM** | `test_pricing_lightgbm` | Обучение, рекомендация цены, прогноз спроса (smoke) |
| **Simulate (P2)** | `test_simulate` | Цикл «история → N дней»: даты, инварианты, rules/regression/lightgbm |

## Не покрыто автотестами

| Область | Комментарий |
|---------|-------------|
| **Streamlit UI** | `views`, `app_entry`, графики, `save_predict_file` |
| **Качество ML на реальных данных** | Сравнение моделей по метрикам на продажах |

