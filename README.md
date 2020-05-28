Build a baseline model for predicting PV yield using NWP (numerical weather predictions), as opposed to satellite imagery. This model is intentionally very simple, so we can get an end-to-end system up and running quickly to interate on.

# Model

The model can be found in the _model_ directory. It is a CSV file with columns for the PV system ID, and the slope and intercept for the regression. Given a value for irradiance ("Downward short-wave radiation flux") from the NWP data at a PV system location, the PV output for the system can be predicted.

## Known limitations
* A linear model is a very big simplification.
* Trained on only a single day's worth of data, at hourly time points.
* Outlier data from the PV systems is included.

# Training

The training data is in a GCP bucket, and is not yet publicly available.

## Install and test

```bash
conda env create -f environment.yml 
conda activate predict_pv_yield_nwp
pip install -e .
pytest -s
```

## Train a model locally

```bash
python predict_pv_yield_nwp/train.py
```
