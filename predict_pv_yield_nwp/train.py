import numpy as np
import os.path
import pandas as pd
from scipy.stats import linregress
import xarray as xr

from predict_pv_yield_nwp.nwp import load_ukv_dataset, UKV1_FILENAME
from predict_pv_yield_nwp.pv import (
    load_pv_systems,
    load_pv_timeseries,
    START_DATE,
    END_DATE,
)


def train(grib_filename: str) -> pd.DataFrame:
    """Train a model to predict per-system PV yield from irradiance."""

    # Load PV timeseries data
    if os.path.exists("data/tmp/pv_timeseries.nc"):  # cached version
        pv_timeseries = xr.load_dataset("data/tmp/pv_timeseries.nc")
    else:
        pv_timeseries = load_pv_timeseries(START_DATE, END_DATE)
        pv_timeseries.to_netcdf("data/tmp/pv_timeseries.nc")

    # Load NWP data
    nwp = load_ukv_dataset(grib_filename)

    # Restrict NWP data to same time slice as PV data
    nwp = nwp.sel(datetime=slice(START_DATE, END_DATE))

    # Sample PV data to times that are in NWP data
    # Note that this only works because the times match (minute boundaries)
    pv_times = pv_timeseries["datetime"].values
    nwp_times = nwp["datetime"].values
    times = list(set(pv_times).intersection(set(nwp_times)))

    pv_subset = pv_timeseries.sel(datetime=times)
    nwp_subset = nwp.sel(datetime=times)

    # Interpolate system locations in the NWP data
    easting = xr.DataArray(pv_subset["easting"].values, dims="system_id")
    northing = xr.DataArray(pv_subset["northing"].values, dims="system_id")
    nwp_interp = nwp_subset.interp(easting=easting, northing=northing)

    # Merge both datasets
    merged = xr.merge([nwp_interp["dswrf"], pv_subset["pv_yield"]])
    merged = merged.load()  # load into memory

    # Stack systems and perform linear regression on each array in the stack
    stacked = merged.to_stacked_array("sys", sample_dims=["system_id"])

    def lr(arr):
        l = len(arr) // 2
        x = arr[:l]  # dswrf
        y = arr[l:]  # pv_yield

        # We need to mask y values that are NaN, since linregress does not do this, see https://stackoverflow.com/a/33550387
        xm = np.ma.masked_array(x, mask=np.isnan(y)).compressed()
        ym = np.ma.masked_array(y, mask=np.isnan(y)).compressed()

        # It's possible that all the values are NaN due to the downsampling to NWP time points.
        # Or there is only one point left.
        # In either case, return zeros, so it can be filtered later on rvalue.
        if len(xm) <= 1:
            return np.array([0, 0, 0])

        res = linregress(xm, ym)
        return np.array([res.slope, res.intercept, res.rvalue])

    lr_result = np.array([lr(arr) for arr in stacked.values])

    # Turn into a dataframe
    df = pd.DataFrame(
        {
            "system_id": pv_subset["system_id"].values,
            "slope": lr_result[:, 0],
            "intercept": lr_result[:, 1],
            "rvalue": lr_result[:, 2],
        }
    )

    # Filter out systems with poor correlation
    df = df.query("rvalue >= 0.6")

    return df


if __name__ == "__main__":
    model = train(UKV1_FILENAME)
    model.to_csv("model/predict_pv_yield_nwp.csv", index=False)
