import os.path

import pytest
from predict_pv_yield_nwp.pv import METADATA_FILENAME, load_pv_systems


@pytest.mark.skipif(not os.path.exists(METADATA_FILENAME), reason="requires test data")
def test_load_pv_systems():
    pv_systems = load_pv_systems()
    coords = list(pv_systems.coords.keys())
    assert "system_id" in coords
    assert "easting" in coords
    assert "northing" in coords

    system = pv_systems.sel(system_id=8200)
    assert system["system_name"] == "Flat 5"
    assert system["easting"] == 534637
    assert system["northing"] == 184127
