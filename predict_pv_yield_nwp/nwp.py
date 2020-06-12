# Read NWP data

# Based on code from https://nbviewer.jupyter.org/github/JackKelly/ng_nwp/blob/master/plot_grib.ipynb

import numpy as np

import xarray as xr

UKV1_FILENAME = "data/NWP/UK_Met_Office/UKV/2019/08/09/NWP_UK_Met_Office_UKV_2019_08_09_201908090000_u1096_ng_umqv_Wholesale2.grib"


def _reshape_data(data: xr.DataArray) -> xr.DataArray:
    KM_TO_M = 1000

    NORTH = 1223 * KM_TO_M
    SOUTH = -185 * KM_TO_M
    WEST = -239 * KM_TO_M
    EAST = 857 * KM_TO_M

    DY = DX = 2 * KM_TO_M

    NORTHING = np.arange(start=SOUTH, stop=NORTH, step=DY, dtype=np.int32)
    EASTING = np.arange(start=WEST, stop=EAST, step=DX, dtype=np.int32)

    NUM_ROWS = len(NORTHING)
    NUM_COLS = len(EASTING)

    # The UKV data is of shape <num_time_steps, num_values> and we want
    # it in shape <num_time_steps, <num_rows>, <num_columns>
    num_time_steps = len(data["step"])
    reshaped = data.values.reshape((num_time_steps, NUM_ROWS, NUM_COLS))

    # The data in the GRIB file starts from the bottom left:
    # 6 7 8
    # 3 4 5
    # 0 1 2
    # Reverse the order of the rows, so the image starts from the top left corner:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    reshaped = reshaped[:, ::-1, :]

    # Get new coords
    coords = data.expand_dims(
        {"northing": NORTHING, "easting": EASTING}, axis=[1, 2]
    ).coords

    # AFAICT, we Xarray doesn't allow us to re-shape data in-place.  Instead,
    # we create a new DataArray with the reshaped data, and the same name & attrs
    # as the source DataArray.
    return xr.DataArray(
        reshaped,
        dims=["step", "northing", "easting"],
        coords=coords,
        name=data.name,
        attrs=data.attrs,
    )


def load_ukv_dataset(grib_filename: str) -> xr.Dataset:
    """Load the UKV data as an xarray dataset"""

    # Open the grib file surface data (for radiation data)
    ds = xr.open_dataset(
        grib_filename,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
    )

    # Reshape data to northing/easting
    # We are only interested in "Downward short-wave radiation flux", or "dswrf"
    ds = xr.Dataset({"dswrf": _reshape_data(ds["dswrf"])})

    # Make valid_time rather than step the dimension, and rename for consistency with PV data
    ds = ds.swap_dims({"step": "valid_time"}).rename({"valid_time": "datetime"})

    return ds
