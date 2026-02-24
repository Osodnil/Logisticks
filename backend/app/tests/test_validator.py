from pathlib import Path

from app.services.validator import validate_customers


def test_validate_customers_has_geocode_warning() -> None:
    out = validate_customers("example/customers_toy.csv")
    assert out["errors"] == []
    assert any(w["message"] == "geocoding_estimated" for w in out["warnings"])
    assert out["cleaned_dataframe"]
    assert Path(out["cleaned_dataframe"]).exists()
