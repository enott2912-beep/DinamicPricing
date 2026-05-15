import numpy as np

from model.math_engine import (
    calc_competitor_1_prices,
    calc_competitor_2_prices,
    calc_demand_regression,
    calc_demand_rules,
)


def test_calc_competitor_1_prices_minimum_one():
    last = np.array([50.0, 100.0])
    base = np.array([55.0, 105.0])
    our = np.array([52.0, 98.0])
    noise = np.zeros(2)
    result = calc_competitor_1_prices(last, base, our, noise)
    assert (result >= 1.0).all()


def test_calc_competitor_2_prices_minimum_one():
    last = np.array([50.0, 100.0])
    base = np.array([55.0, 105.0])
    anchor = np.array([48.0, 95.0])
    noise = np.zeros(2)
    chaotic = np.array([False, True])
    result = calc_competitor_2_prices(last, base, anchor, noise, chaotic)
    assert (result >= 1.0).all()


def test_calc_demand_rules_non_negative():
    our = np.array([80.0, 90.0, 100.0])
    comp = np.array([82.0, 88.0, 95.0])
    base_p = np.array([85.0, 85.0, 85.0])
    base_s = np.array([300.0, 300.0, 300.0])
    elast = np.array([2.0, 2.0, 2.0])
    noise = np.zeros(3)
    result = calc_demand_rules(our, comp, base_p, base_s, elast, noise)
    assert (result >= 0).all()


def test_calc_demand_regression_matches_input_length():
    prices = np.array([80.0, 90.0, 100.0])
    a = np.full(3, 400.0)
    b = np.full(3, 3.0)
    noise = np.zeros(3)
    result = calc_demand_regression(prices, a, b, noise)
    assert len(result) == len(prices)


def test_calc_demand_rules_higher_price_lower_demand():
    our_low = np.array([80.0])
    our_high = np.array([100.0])
    comp = np.array([85.0])
    base_p = np.array([85.0])
    base_s = np.array([300.0])
    elast = np.array([2.0])
    noise = np.zeros(1)
    low_sales = calc_demand_rules(our_low, comp, base_p, base_s, elast, noise)[0]
    high_sales = calc_demand_rules(our_high, comp, base_p, base_s, elast, noise)[0]
    assert high_sales < low_sales


def test_calc_competitor_1_increases_when_our_price_increases():
    last = np.array([90.0, 90.0])
    base = np.array([90.0, 90.0])
    our_low = np.array([80.0, 80.0])
    our_high = np.array([110.0, 110.0])
    noise = np.zeros(2)
    p_low = calc_competitor_1_prices(last, base, our_low, noise)
    p_high = calc_competitor_1_prices(last, base, our_high, noise)
    assert p_high[0] >= p_low[0]
