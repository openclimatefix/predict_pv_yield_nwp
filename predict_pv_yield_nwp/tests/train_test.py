import os.path
import pytest

from predict_pv_yield_nwp.train import train
from predict_pv_yield_nwp.nwp import UKV1_FILENAME


@pytest.mark.skipif(not os.path.exists(UKV1_FILENAME), reason="requires test data")
def test_train():
    model = train(UKV1_FILENAME)

    columns = model.columns

    assert "system_id" in columns
    assert "slope" in columns
    assert "intercept" in columns
