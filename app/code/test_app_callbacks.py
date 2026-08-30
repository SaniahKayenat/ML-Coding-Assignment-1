"""
Unit tests for the prediction logic
no browser or running server needed.

Run with:  cd app/code && pytest
"""

import main  
from pages import predict


def test_predict_price_returns_a_positive_number():
    price = predict.predict_price(
        brand="Maruti", year=2019, km_driven=35000, fuel="Petrol",
        seller_type="Individual", transmission="Manual", owner=1,
        mileage=21.5, engine=1197, max_power=82, seats=5,
    )
    assert isinstance(price, float)
    assert price > 0


def test_predict_price_handles_missing_optional_fields():
    # mileage / engine / max_power / seats all left out entirely
    price = predict.predict_price(
        brand="Toyota", year=2018, km_driven=45000, fuel="Diesel",
        seller_type="Individual", transmission="Manual", owner=1,
    )
    assert isinstance(price, float)
    assert price > 0


def test_newer_car_predicts_higher_than_an_older_otherwise_identical_car():
    newer = predict.predict_price(
        brand="Maruti", year=2020, km_driven=30000, fuel="Petrol",
        seller_type="Individual", transmission="Manual", owner=1, max_power=82,
    )
    older = predict.predict_price(
        brand="Maruti", year=2005, km_driven=30000, fuel="Petrol",
        seller_type="Individual", transmission="Manual", owner=1, max_power=82,
    )
    assert newer > older


def test_more_km_driven_predicts_lower_or_equal_price():
    low_km = predict.predict_price(
        brand="Hyundai", year=2017, km_driven=10000, fuel="Petrol",
        seller_type="Individual", transmission="Manual", owner=1, max_power=88,
    )
    high_km = predict.predict_price(
        brand="Hyundai", year=2017, km_driven=150000, fuel="Petrol",
        seller_type="Individual", transmission="Manual", owner=1, max_power=88,
    )
    assert low_km >= high_km
