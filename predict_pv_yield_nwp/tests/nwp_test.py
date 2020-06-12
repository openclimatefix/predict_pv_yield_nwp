import os.path

import pytest
from predict_pv_yield_nwp.nwp import UKV1_FILENAME, load_ukv_dataset


@pytest.mark.skipif(not os.path.exists(UKV1_FILENAME), reason="requires test data")
def test_load_ukv_dataset():
    ukv = load_ukv_dataset(UKV1_FILENAME)

    coords = list(ukv.coords.keys())
    assert "datetime" in coords
    assert "easting" in coords
    assert "northing" in coords

    variables = list(ukv.variables.keys())
    assert "dswrf" in variables
