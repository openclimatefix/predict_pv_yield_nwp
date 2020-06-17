# Read PV metadata and timeseries data

# Based on code in https://github.com/openclimatefix/pvoutput
# E.g. https://nbviewer.jupyter.org/github/openclimatefix/pvoutput/blob/master/examples/analyse_PV_data_for_9th_Aug_2019.ipynb

import cartopy.crs as ccrs
import numpy as np
import pandas as pd

import xarray as xr

METADATA_FILENAME = "data/PV/PVOutput.org/UK_PV_metadata.csv"
PV_STATS_FILENAME = "data/PV/PVOutput.org/UK_PV_stats.csv"
TIMESERIES_FILENAME = "data/PV/PVOutput.org/UK_PV_timeseries_batch.nc"

START_DATE = "2019-08-09"
END_DATE = "2019-08-09"


def load_pv_systems(
    metadata_filename: str = METADATA_FILENAME,
    stats_filename: str = PV_STATS_FILENAME,
    timeseries_filename: str = TIMESERIES_FILENAME,
) -> xr.Dataset:
    """Load metadata about PV systems"""

    # Load metadata
    pv_metadata = pd.read_csv(metadata_filename, index_col="system_id")

    # Load stats
    pv_stats = pd.read_csv(
        stats_filename,
        index_col="system_id",
        parse_dates=["actual_date_from", "actual_date_to", "record_efficiency_date"],
    )

    # Join
    pv_systems = pv_metadata.join(
        pv_stats[["actual_date_from", "actual_date_to", "outputs"]], how="left"
    )

    # Filter out systems with only a few outputs, and with no location
    pv_systems_filtered = pv_systems.query(
        "status_interval_minutes <= 60 and outputs > 100"
    )
    pv_systems_filtered = pv_systems_filtered.dropna(subset=["latitude", "longitude"])

    # Restrict to systems that have timeseries data
    system_ids = _get_system_ids_dataframe_from_timeseries(timeseries_filename)
    pv_systems_filtered = pv_systems_filtered.join(system_ids, how="inner")

    # Retain salient columns
    pv_systems_filtered = pv_systems_filtered[["system_name", "latitude", "longitude"]]

    # Convert to xarray
    ds = xr.Dataset.from_dataframe(pv_systems_filtered)

    # Convert latitude/longitude to easting/northing
    ds = _transform_pv_systems(ds)

    return ds


def _get_system_ids_dataframe_from_timeseries(
    timeseries_filename: str = TIMESERIES_FILENAME,
) -> pd.DataFrame:
    """Get all the PV system IDs from the timeseries file"""
    ds = xr.open_dataset(timeseries_filename)
    system_ids = [int(x) for x in list(ds.data_vars.keys())]
    df = pd.DataFrame({"system_id": system_ids})
    df = df.set_index("system_id")
    return df


def _transform_pv_systems(pv_systems: xr.Dataset) -> xr.Dataset:
    """Transform the system locations into the same coordinate system used by UKV"""

    system_latitudes, system_longitudes = (
        pv_systems["latitude"].values,
        pv_systems["longitude"].values,
    )

    wgs84 = ccrs.Geodetic()
    ukv_crs = ccrs.OSGB(approx=False)
    locs = ukv_crs.transform_points(
        src_crs=wgs84,
        x=np.asanyarray(system_longitudes),
        y=np.asanyarray(system_latitudes),
    )[:, :-1]

    new_coords = {
        "easting": (["system_id"], locs[:, 0].astype("int32")),
        "northing": (["system_id"], locs[:, 1].astype("int32")),
    }
    return pv_systems.assign_coords(new_coords)


# This is unused, but a useful check
def _transform_pv_systems_pyproj(pv_systems: xr.Dataset) -> xr.Dataset:
    """Transform the system locations into the same coordinate system used by UKV, using pyproj"""
    import pyproj

    system_latitudes, system_longitudes = (
        pv_systems["latitude"].values,
        pv_systems["longitude"].values,
    )

    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:27700", always_xy=True)
    locs = transformer.transform(
        np.asanyarray(system_longitudes), np.asanyarray(system_latitudes)
    )
    print(locs)

    new_coords = {
        "easting": (["system_id"], locs[0]),
        "northing": (["system_id"], locs[1]),
    }
    return pv_systems.assign_coords(new_coords)


def load_pv_timeseries(
    start_date: str,
    end_date: str,
    metadata_filename: str = METADATA_FILENAME,
    stats_filename: str = PV_STATS_FILENAME,
    timeseries_filename: str = TIMESERIES_FILENAME,
) -> xr.Dataset:
    """Load the PV timeseries as an xarray dataset, restricted to a given time range, and including location metadata."""

    ds = xr.open_dataset(timeseries_filename)

    # Subset to given time range
    subset = ds.sel(datetime=slice(start_date, end_date))

    # Drop systems with no readings during this time
    # I couldn't see how to do this with xarray, see https://stackoverflow.com/questions/52553925/python-xarray-remove-coordinates-with-all-missing-variables
    df = subset.to_dataframe()
    df = df.dropna(axis=1, how="all")

    # Restrict to systems that are in the intersection of those in PV metadata and PV timeseries
    pv_df = load_pv_systems(
        metadata_filename, stats_filename, timeseries_filename
    ).to_dataframe()
    pv_metadata_system_ids = pv_df.index.tolist()  # indexed by system_id
    timeseries_system_ids = [int(system_id) for system_id in df.columns.tolist()]
    system_ids = list(
        set(pv_metadata_system_ids).intersection(set(timeseries_system_ids))
    )
    system_id_columns = [str(system_id) for system_id in system_ids]
    df = df[system_id_columns]

    # Reshape table into tall and narrow form - this avoids one data variable per system in xarray
    df["datetime"] = df.index
    df = pd.melt(df, id_vars=["datetime"], var_name="system_id", value_name="pv_yield")
    df = df.astype({"system_id": "int64"})
    df = df.set_index(["system_id", "datetime"])

    # Convert back to xarray
    ds = xr.Dataset.from_dataframe(df)

    # Add lat/long and easting/northing coordinates by doing a pandas lookup for each system
    new_coords = {
        "latitude": (
            ["system_id"],
            pv_df.lookup(system_ids, ["latitude"] * len(system_ids)),
        ),
        "longitude": (
            ["system_id"],
            pv_df.lookup(system_ids, ["longitude"] * len(system_ids)),
        ),
        "easting": (
            ["system_id"],
            pv_df.lookup(system_ids, ["easting"] * len(system_ids)),
        ),
        "northing": (
            ["system_id"],
            pv_df.lookup(system_ids, ["northing"] * len(system_ids)),
        ),
    }
    ds = ds.assign_coords(new_coords)

    return ds


if __name__ == "__main__":
    pv_timeseries = load_pv_timeseries(START_DATE, END_DATE)
    print(pv_timeseries)

    pv_timeseries.to_netcdf("data/tmp/pv_timeseries.nc")
