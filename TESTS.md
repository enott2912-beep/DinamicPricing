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
| `test_rules_engine.py` | `model/rules_engine.py` | 10 |
| `test_pricing_regression.py` | `model/pricing.py` (линейная регрессия, CSV SKU) | 10 |
| `test_pricing_forecast.py` | `model/pricing.py` (прогноз, правила) | 4 |
| `test_custom_csv_products.py` | `model/pricing.py`, `model/analytics.py` (свой CSV) | 3 |
| `test_data_validation.py` | `ui/data_manager.py` → `_validate_loaded_data` | 5 |
| `test_pricing_lightgbm.py` | `model/pricing.py` (LightGBM) | 9 |
| `test_simulate.py` | `model/pricing.py` → `simulate` | 16 |
| `test_external_dataset_sanity.py` | `model/pricing.py` (регрессия) на Retail Price | 4 |
| `test_lgbm_thresholds_scale_invariant.py` | `model/pricing.py` (LightGBM) на Avocado | 3 |
| `test_is_oos_parsing.py` | `model/utils.py`, `ui/data_manager.py` | 26 |
| **Итого** | | **104** |

### `conftest.py` (не тесты)

| Фикстура | Назначение |
|----------|------------|
| `minimal_sales_df` | 8 дней истории по SKU «Молоко» с согласованными revenue/profit |
| `lgbm_training_df` | 70 дней с вариативными ценами — достаточно для обучения LightGBM. **Изменено:** убран линейный тренд цены (`+ np.linspace(-4, 4, n)`), который вызывал экстраполяцию между train и holdout и давал ложно плохие метрики. |
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
| 21 | `test_rules_engine.py` | `test_save_rules_without_session_writes_file` | `RuleEngine._save_rules_to_file` | Без `session_id` правила пишутся в `rules_path`. |
| 22 | `test_rules_engine.py` | `test_save_rules_session_db_failure_does_not_touch_global_file` | `RuleEngine._save_rules_to_session` | Сбой БД при сохранении сессии → `RuntimeError`, глобальный `rules.json` не меняется. |
| 23 | `test_rules_engine.py` | `test_load_rules_session_db_failure_raises` | `RuleEngine.load_rules` | Сбой чтения из БД → `RuntimeError`, без тихого fallback на файл. |
| 24 | `test_rules_engine.py` | `test_save_rules_session_success_uses_db` | `RuleEngine._save_rules_to_session` | Успешное сохранение сессии → данные в SQLite, глобальный файл не трогается. |
| 25 | `test_pricing_regression.py` | `test_fit_linear_sales_vs_price_insufficient_data` | `_fit_linear_sales_vs_price` | Меньше 3 точек → модель ненадёжна, fallback-цена. |
| 26 | `test_pricing_regression.py` | `test_fit_linear_sales_vs_price_negative_slope_reliable` | `_fit_linear_sales_vs_price` | Sales падают с ценой → B>0, `reliable=True`, оптимум ≥ 1. |
| 27 | `test_pricing_regression.py` | `test_fit_linear_optimal_price_near_formula` | `_fit_linear_sales_vs_price` | Оптимальная цена близка к (A+B*COGS)/(2B) с учётом клиппинга |
| 28 | `test_pricing_regression.py` | `test_fit_regression_filters_oos` | `fit_regression` | Строки `is_oos=True` с аномальными sales не портят обучение. |
| 29 | `test_pricing_regression.py` | `test_fit_regression_filters_zero_sales` | `fit_regression` | День с `sales=0` исключается из обучения. |
| 30 | `test_pricing_regression.py` | `test_predict_sales_regression_formula` | `predict_sales_regression` | Спрогноз = max(0, round(A-B*Price)) без шума |
| 31 | `test_pricing_regression.py` | `test_predict_sales_regression_with_noise` | `predict_sales_regression` | То же с добавлением шума в формулу. |
| 32 | `test_pricing_regression.py` | `test_products_in_dataframe_order` | `products_in_dataframe` | Уникальные SKU из df, стабильная сортировка по имени. |
| 33 | `test_pricing_regression.py` | `test_products_in_dataframe_custom_sku` | `products_in_dataframe` | Произвольное имя товара (не из `PRODUCTS`) попадает в список. |
| 34 | `test_pricing_regression.py` | `test_infer_entity_params_from_history` | `infer_entity_params` | Параметры base_price, base_sales, elasticity, cogs оцениваются по истории. |
| 35 | `test_pricing_forecast.py` | `test_forecast_with_regression_params` | `forecast` | С коэффициентами (A,B): sales, profit и growth_pct согласованы. |
| 36 | `test_pricing_forecast.py` | `test_forecast_without_regression_uses_products` | `forecast` | Без регрессии и без `product_params` — спрос из констант `PRODUCTS`. |
| 37 | `test_pricing_forecast.py` | `test_forecast_without_regression_uses_inferred_params` | `forecast` | Без регрессии с `product_params` — спрос по эвристике из истории. |
| 38 | `test_pricing_forecast.py` | `test_apply_rules_delegates_to_engine` | `apply_rules` | Контекст из строки DataFrame передаётся в движок правил. |
| 39 | `test_custom_csv_products.py` | `test_simulate_custom_product_no_keyerror` | `simulate` | SKU вне `PRODUCTS` симулируется без `KeyError`, появляются будущие дни. |
| 40 | `test_custom_csv_products.py` | `test_recommendations_custom_products` | `get_recommendations_all_products` | Портфельные рекомендации не пустые для произвольного SKU. |
| 41 | `test_custom_csv_products.py` | `test_forecast_with_inferred_params` | `forecast` + `infer_entity_params` | Прогноз для чужого SKU с параметрами из истории. |
| 42 | `test_data_validation.py` | `test_validate_loaded_data_valid_minimal` | `_validate_loaded_data` | Минимально корректная строка CSV → DataFrame, без ошибки UI. |
| 43 | `test_data_validation.py` | `test_validate_loaded_data_missing_column` | `_validate_loaded_data` | Нет обязательной колонки → `None`, вызов `st.error`. |
| 44 | `test_data_validation.py` | `test_validate_loaded_data_revenue_mismatch` | `_validate_loaded_data` | `revenue` ≠ `sales × our_price` → отклонение. |
| 45 | `test_data_validation.py` | `test_validate_loaded_data_negative_price` | `_validate_loaded_data` | Отрицательная `our_price` → отклонение. |
| 46 | `test_data_validation.py` | `test_validate_loaded_data_bad_date` | `_validate_loaded_data` | Неразбираемая дата → отклонение. |
| 47 | `test_pricing_lightgbm.py` | `test_lgbm_data_warnings_short_history` | `_lgbm_data_warnings` | Короткая история → предупреждение о малом числе наблюдений. |
| 48 | `test_pricing_lightgbm.py` | `test_build_lgbm_training_frame_filters_oos` | `_build_lgbm_training_frame` | OOS и нулевые sales не попадают в обучающую выборку. |
| 49 | `test_pricing_lightgbm.py` | `test_fit_lightgbm_short_history_unreliable` | `fit_lightgbm_sales_model` | 8 дней → модель не обучается, `reliable=False`. |
| 50 | `test_pricing_lightgbm.py` | `test_fit_lightgbm_empty_after_clean` | `fit_lightgbm_sales_model` | Пустой df после очистки → модель `None`. |
| 51 | `test_pricing_lightgbm.py` | `test_fit_lightgbm_reliable_on_rich_history` | `fit_lightgbm_sales_model` | 70 дней с вариацией цен → модель обучена, `reliable=True`. |
| 52 | `test_pricing_lightgbm.py` | `test_recommend_price_lightgbm_with_pack_respects_step_limit` | `recommend_price_lightgbm_with_pack` | Рекомендованная цена в пределах дневного лимита ±2%. |
| 53 | `test_pricing_lightgbm.py` | `test_recommend_price_lightgbm_fallback_when_unreliable` | `recommend_price_lightgbm` | Слабые данные → цена не меняется (hold последней). |
| 54 | `test_pricing_lightgbm.py` | `test_predict_sales_lightgbm_non_negative` | `predict_sales_lightgbm_with_pack` | Прогноз спроса ≥ 0 при обученной модели. |
| 55 | `test_pricing_lightgbm.py` | `test_predict_sales_fallback_when_no_model` | `predict_sales_lightgbm_with_pack` | Без модели → среднее sales за 7 дней. |
| 56 | `test_simulate.py` | `test_simulate_appends_n_steps_days` | `simulate` | После n_steps уникальных дат стало ровно на столько больше |
| 57 | `test_simulate.py` | `test_simulate_future_dates_after_history_max` | `simulate` | Все новые даты строго позже конца истории |
| 58 | `test_simulate.py` | `test_simulate_output_schema` | `simulate` | В будущих строках есть цены, спрос, выручка, прибыль, конкуренты |
| 59 | `test_simulate.py` | `test_simulate_invariants` | `simulate` | our_price >= 1, sales >= 0, revenue >= 0 на прогнозных днях |
| 60 | `test_simulate.py` | `test_simulate_accounting` | `simulate` | revenue = sales*price, profit = revenue - sales*cogs |
| 61 | `test_simulate.py` | `test_simulate_reproducible_rules` | `simulate` | Два прогона method=rules с одним df -> идентичный будущий хвост (SEED) |
| 62 | `test_simulate.py` | `test_simulate_empty_or_no_products` | `simulate` | Пустой df -> без строк; произвольный SKU -> n_steps новых дней без падения |
| 63 | `test_simulate.py` | `test_simulate_target_product_filter` | `simulate` | target_product ограничивает прогноз одним SKU |
| 64 | `test_simulate.py` | `test_simulate_row_count` | `simulate` | Число прогнозных строк = n_steps * число сущностей |
| 65 | `test_simulate.py` | `test_simulate_regression_completes` | `simulate` | method=regression завершается на 21 дне истории |
| 66 | `test_simulate.py` | `test_simulate_regression_daily_price_step_limit` | `simulate` | Шаг цены между днями в пределах max_daily_price_change_pct |
| 67 | `test_simulate.py` | `test_simulate_regression_train_window_clipped` | `simulate` | Окно 999 дней при короткой истории не роняет симуляцию |
| 68 | `test_simulate.py` | `test_simulate_rules_completes` | `simulate` | method=rules smoke на 5 шагов |
| 69 | `test_simulate.py` | `test_simulate_rules_with_patched_engine` | `simulate` | С моком правила price*1.1 первая будущая цена соответствует правилу |
| 70 | `test_simulate.py` | `test_simulate_lightgbm_completes` | `simulate` | method=lightgbm на длинной истории - без падения, sales >= 0 |
| 71 | `test_simulate.py` | `test_simulate_lightgbm_short_history` | `simulate` | Короткая история + lightgbm - fallback, без падения |
| 72 | `test_external_dataset_sanity.py` | `test_external_dataset_loads_with_expected_columns` | (загрузка) | Датасет Retail Price: загрузка, колонки, >= 30 SKU, >= 300 строк |
| 73 | `test_external_dataset_sanity.py` | `test_reliability_rate_is_not_inflated` | `_fit_linear_sales_vs_price` | Доля reliable по новому критерию (R2 >= 0.25) не превышает 40% |
| 74 | `test_external_dataset_sanity.py` | `test_confidence_weight_is_fractional_for_weak_fits` | `_ols_fit_diagnostics`, `_confidence_weight` | Для слабого отриц. коэффициента вес доверия строго между 0 и 1 |
| 75 | `test_external_dataset_sanity.py` | `test_simulate_regression_runs_on_external_data_without_crashing` | `simulate` | method=regression на шумных внешних данных - без исключений, цены > 0 |
| 76 | `test_lgbm_thresholds_scale_invariant.py` | `test_avocado_fixture_has_low_absolute_but_real_relative_variability` | (sanity) | Фикстура avocado: маленький abs std (< 1.0), но CV > LGBM_MIN_PRICE_CV |
| 77 | `test_lgbm_thresholds_scale_invariant.py` | `test_lgbm_price_variability_threshold_is_scale_invariant` | `_lgbm_data_warnings` | Ни одна сущность не получает false positive предупреждение о диапазоне цен |
| 78 | `test_lgbm_thresholds_scale_invariant.py` | `test_lgbm_fits_and_runs_holdout_validation_on_avocado_data` | `fit_lightgbm_sales_model` | Модель обучается на avocado, holdout R2 и MAE ratio не None |
| 79 | `test_is_oos_parsing.py` | `test_parse_is_oos_value` (param x16) | `parse_is_oos_value` | 16 форматов is_oos (False, True, 0, 1, "false", "Да", "отсутствует") -> bool |
| 80 | `test_is_oos_parsing.py` | `test_parse_is_oos_series_mixed_strings` | `parse_is_oos_series` | Серия со смешанными строковыми значениями -> корректный список bool |
| 81 | `test_is_oos_parsing.py` | `test_parse_is_oos_value_unknown_raises` | `parse_is_oos_value` | Неизвестное значение -> ValueError |
| 82 | `test_is_oos_parsing.py` | `test_validate_loaded_data_is_oos_false_strings` (param x4) | `_validate_loaded_data` | is_oos = "Нет"/"no"/"0"/"false" -> False, без st.error |
| 83 | `test_is_oos_parsing.py` | `test_validate_loaded_data_is_oos_true_strings` (param x4) | `_validate_loaded_data` | is_oos = "Да"/"yes"/"1"/"отсутствует" -> True, без st.error |

**Итого: 104 теста** (P0: 24, P1: 19, Custom CSV: 3, LightGBM: 9, Simulate: 16, External sanity: 4, Scale-invariant: 3, OOS parsing: 26). Если `lightgbm` не установлен, 11 тестов пропускаются (`pytest.skip`).

---

## Группировка по приоритету

| Приоритет | Файлы | Фокус |
|-----------|--------|--------|
| **P0** | `test_math_engine`, `test_rules_validator`, `test_rules_engine`, `test_is_oos_parsing` | Формулы спроса и конкурентов; валидация Rule Engine; сессионное сохранение правил; разбор разных форматов is_oos |
| **P1** | `test_pricing_regression`, `test_pricing_forecast`, `test_data_validation` | Линейная регрессия и прогноз; валидация загружаемого CSV |
| **External datasets** | `test_external_dataset_sanity`, `test_lgbm_thresholds_scale_invariant` | Валидация на реальных датасетах (Retail Price - регрессия, Avocado - LightGBM); регрессионные барьеры |
| **Custom CSV** | `test_custom_csv_products` | Произвольные SKU: рекомендации, симуляция, `infer_entity_params` |
| **LightGBM** | `test_pricing_lightgbm` | Обучение, рекомендация цены, прогноз спроса (smoke) |
| **Simulate (P2)** | `test_simulate` | Цикл "история -> N дней": даты, инварианты, rules/regression/lightgbm |

