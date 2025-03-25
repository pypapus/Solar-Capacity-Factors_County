"""
Download solar and wind resource data at the county level for the Eastern Interconnect
using NREL's NSRDB & WIND Toolkit. Compute capacity factors via PySAM (PVWatts v8, Windpower).

Author: Papa Yaw Owusu-Obeng
Date: 2025-03-25
"""

import argparse
import os
import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as pv
import PySAM.Windpower as wp
import geopandas as gpd
from shapely.geometry import Point
import time
import datetime
import warnings
from urllib.error import HTTPError

warnings.filterwarnings("ignore")

# Path where script will write output and store downloaded data
local_path = os.path.dirname(os.path.abspath(__file__))

#################################
# 1) DEFINE EASTERN INTERCONNECT STATES (approx)
#################################
# This is an approximate list of states commonly considered part of the Eastern Interconnect.
# Real boundaries can differ (e.g. part of TX is ERCOT, part of OK is in the Western Interconnect, etc.)
EASTERN_INTERCONNECT_STATES = [
    "AL","AR","CT","DE","FL","GA","IA","IL","IN","KY","LA","MA","MD","ME","MI","MN","MO",
    "MS","NC","ND","NE","NH","NJ","NY","OH","PA","RI","SC","SD","TN","VA","VT","WI","WV","KS", "OK"
    # Possibly partially: KS, OK, TX can have areas in other interconnects
]

#################################
# 2) ARGUMENT PARSING
#################################
parser = argparse.ArgumentParser(
    description="Download wind and solar resource data at the county level for the Eastern Interconnect, then use PySAM to compute hourly CF."
)

parser.add_argument("--year", type=int, choices=range(2007,2015), required=True,
                    help="Data year (2007-2014).")
parser.add_argument("--api-key", type=str, required=True,
                    help="NREL API Key (get one at https://developer.nrel.gov/signup/)")
parser.add_argument("--email", type=str, required=True,
                    help="Email address (required by NREL API).")

parser.add_argument("--counties_shapefile", type=str, required=True,
                    help="Path to the US counties shapefile (nationwide).")

parser.add_argument("--save_resource", action="store_true",
                    help="If set, retains downloaded NSRDB CSV and WTK SRW files on disk.")
parser.add_argument("--verbose", action="store_true",
                    help="Print debugging messages.")
parser.add_argument("--sleep_seconds", type=float, default=1.0,
                    help="Seconds to sleep between each location request (avoid rate limits).")

args = parser.parse_args()

#################################
# 3) HELPER FUNCTIONS
#################################

def getSolarResourceData(year, lat, lon):
    """
    Downloads (or loads) solar resource data from NSRDB for lat, lon in the specified year.
    Returns a dictionary suitable for PySAM.
    """
    # Make sure we have a local directory for resource data
    resource_dir = os.path.join(local_path, "resourceData")
    if not os.path.exists(resource_dir):
        os.makedirs(resource_dir)

    out_csv = os.path.join(resource_dir, f"{lat}_{lon}_nsrdb.csv")

    if os.path.exists(out_csv):
        # Already downloaded
        solarResource = pd.read_csv(out_csv)
    else:
        # Download from NSRDB PSM3
        url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
        params = {
            "api_key": args.api_key,
            "email": args.email,
            "wkt": f"POINT({lon}+{lat})",  # note the plus sign for lat-lon in wkt param
            "names": year,
            "utc": "true"
        }
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        download_url = f"{url}?{param_str}"
        try:
            solarResource = pd.read_csv(download_url)
        except HTTPError as e:
            raise e

        # Save locally
        solarResource.to_csv(out_csv, index=False)

    # Row 0 = metadata names, Row 1 = units, so the actual data starts at row 2
    meta = solarResource.head(1)
    lat_meta = meta.at[0, "Latitude"]
    lon_meta = meta.at[0, "Longitude"]
    tz = meta.at[0, "Time Zone"]
    elev = meta.at[0, "Elevation"]

    # 2nd line => column names
    new_cols = solarResource.iloc[1].to_list()

    # Drop first two lines
    solarResource = solarResource.drop([0, 1]).reset_index(drop=True)
    solarResource.columns = new_cols

    # Build dictionary for PySAM
    sol_dict = {
        "lat": float(lat_meta),
        "lon": float(lon_meta),
        "tz": float(tz),
        "elev": float(elev),
        "year": solarResource["Year"].astype(float).tolist(),
        "month": solarResource["Month"].astype(float).tolist(),
        "day": solarResource["Day"].astype(float).tolist(),
        "hour": solarResource["Hour"].astype(float).tolist(),
        "minute": solarResource["Minute"].astype(float).tolist(),
        "dn": solarResource["DNI"].astype(float).tolist(),
        "df": solarResource["DHI"].astype(float).tolist(),
        "gh": solarResource["GHI"].astype(float).tolist(),
        "wspd": solarResource["Wind Speed"].astype(float).tolist(),
        "tdry": solarResource["Temperature"].astype(float).tolist()
    }

    return sol_dict


def getSolarCF(solarResourceData):
    """
    Run PVWatts with the solar resource data, returning an hourly capacity factor timeseries.
    """
    s = pv.default("PVWattsNone")
    s.SolarResource.solar_resource_data = solarResourceData

    # Basic system params
    s.SystemDesign.array_type = 2       # 2 => 1-axis
    s.SystemDesign.azimuth = 180
    s.SystemDesign.tilt = abs(solarResourceData["lat"])
    nameplate_kW = 1000  # 1 MW DC
    s.SystemDesign.system_capacity = nameplate_kW
    s.SystemDesign.dc_ac_ratio = 1.1
    s.SystemDesign.inv_eff = 96
    s.SystemDesign.losses = 14

    s.execute()

    # Convert AC in W to capacity factor
    # AC power timeseries -> s.Outputs.ac
    ac_watts = np.array(s.Outputs.ac)    # each entry in W
    solar_cf = ac_watts / (nameplate_kW * 1000.0)
    return solar_cf


def getWindSRW(year, lat, lon):
    """
    Download or load Wind Toolkit SRW file for lat, lon, year. Returns (filename, IEC_class).
    """
    resource_dir = os.path.join(local_path, "resourceData")
    if not os.path.exists(resource_dir):
        os.makedirs(resource_dir)

    out_srw = os.path.join(resource_dir, f"{lat}_{lon}_wtk.srw")

    if not os.path.exists(out_srw):
        url = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-srw-download"
        params = {
            "api_key": args.api_key,
            "email": args.email,
            "lat": lat,
            "lon": lon,
            "hubheight": 100,
            "year": year,
            "utc": "true"
        }
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        download_url = f"{url}?{param_str}"

        wtk_data = pd.read_csv(download_url)
        wtk_data.to_csv(out_srw, index=False)

    # Determine IEC class based on median wind speed
    # We'll skip the lines that contain units or meta info
    # Usually row 2 is data start, row 0 is times
    df_speed = pd.read_csv(out_srw, skiprows=[0, 1, 3, 4], usecols=["Speed"])
    median_ws = np.median(df_speed["Speed"].values)
    if median_ws >= 9:
        iec_class = 1
    elif median_ws >= 8:
        iec_class = 2
    else:
        iec_class = 3

    return out_srw, iec_class


def getWindCF(windSRW, iecClass, powerCurve, save_resource=False):
    """
    Use PySAM Windpower to compute capacity factor from an SRW file.
    Removes the SRW file if save_resource is False.
    """
    d = wp.default("WindPowerNone")

    # Build the composite curve (IEC class)
    speeds = powerCurve["Wind Speed"]
    powerout = powerCurve[f"Composite IEC Class {iecClass}"]

    d.Resource.wind_resource_filename = windSRW
    d.Resource.wind_resource_model_choice = 0
    d.Turbine.wind_turbine_powercurve_powerout = powerout
    d.Turbine.wind_turbine_powercurve_windspeeds = speeds
    d.Turbine.wind_turbine_rotor_diameter = 90
    d.Turbine.wind_turbine_hub_ht = 100

    nameplate_kW = 1500.0
    d.Farm.system_capacity = nameplate_kW
    d.Farm.wind_farm_wake_model = 0
    d.Farm.wind_farm_xCoordinates = np.array([0])
    d.Farm.wind_farm_yCoordinates = np.array([0])

    d.execute()

    wind_cf = np.array(d.Outputs.gen) / nameplate_kW

    # If not saving resource, remove the .srw
    if not save_resource:
        try:
            os.remove(windSRW)
        except:
            pass

    return wind_cf


def filter_eastern_counties(shp_path):
    """
    Load the counties shapefile, filter it to states in the Eastern Interconnect,
    then return a list of (lat, lon) from county centroids.
    """
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs("EPSG:4326")

    # Shapefile typically has a column like 'STATEFP' or 'STUSPS' for the state abbreviation
    # Many county shapefiles have 'STATEFP' as a 2-digit FIPS code, or 'STUSPS' as the 2-letter abbreviation.
    # We'll try 'STUSPS' first. If your file is different, adjust accordingly.
    col_candidates = [c.upper() for c in gdf.columns]
    if "STUSPS" in col_candidates:
        st_col = [c for c in gdf.columns if c.upper() == "STUSPS"][0]
        # Filter to counties that have STUSPS in our Eastern states list
        # Some shapefiles store DC as 'DC'; some store it as '11'; so watch out for special cases.
        # We'll just filter if the state is in EASTERN_INTERCONNECT_STATES
        gdf_filtered = gdf[gdf[st_col].isin(EASTERN_INTERCONNECT_STATES)]
    else:
        raise ValueError("Could not find STUSPS column in county shapefile. Please adjust the script to match your shapefile's state column.")

    # Return county centroids
    coords = []
    for _, row in gdf_filtered.iterrows():
        centroid = row.geometry.centroid
        coords.append( (centroid.y, centroid.x) )  # (lat, lon)

    return coords

#################################
# 4) MAIN FUNCTION
#################################
def main():
    start = datetime.datetime.now()

    # Create output directory
    out_dir = os.path.join(local_path, "output")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create a file name for this run
    # e.g. "eastern_county_solar_cf_2012.csv"
    solar_filename = os.path.join(out_dir, f"eastern_county_solar_cf_{args.year}.csv")
    wind_filename  = os.path.join(out_dir, f"eastern_county_wind_cf_{args.year}.csv")

    # We'll store partial progress
    solar_df = pd.DataFrame()
    wind_df = pd.DataFrame()

    # If partial data from a prior run exist, load them
    if os.path.exists(solar_filename) and os.path.exists(wind_filename):
        solar_df = pd.read_csv(solar_filename, index_col=0)
        wind_df = pd.read_csv(wind_filename, index_col=0)

    # 1) Get all county centroids for Eastern Interconnect
    coords = filter_eastern_counties(args.counties_shapefile)
    print(f"Found {len(coords)} counties in Eastern Interconnect states (approx).")

    # 2) Filter out coords we already processed
    completed_coords = set()
    if not solar_df.empty and not wind_df.empty:
        # Our columns are "lat lon"
        done_cols = solar_df.columns.intersection(wind_df.columns)
        for col in done_cols:
            lat_str, lon_str = col.split(" ")
            completed_coords.add( (float(lat_str), float(lon_str)) )

    # 3) Load the wind power curves (must be in local_path/powerCurves/powerCurves.csv)
    powercurves_path = os.path.join(local_path, "powerCurves", "powerCurves.csv")
    if not os.path.exists(powercurves_path):
        raise FileNotFoundError(f"Power curve CSV not found: {powercurves_path}")
    powerCurve = pd.read_csv(powercurves_path)

    # 4) MAIN LOOP
    to_process = [c for c in coords if c not in completed_coords]
    if len(to_process) == 0:
        print("All counties already processed. Exiting.")
        return

    print(f"Need to process {len(to_process)} counties...")
    for i, (lat, lon) in enumerate(to_process):
        if args.verbose:
            print(f"Processing {i+1}/{len(to_process)}: (lat={lat}, lon={lon})")

        # Attempt the calls
        try:
            # Solar
            solar_data = getSolarResourceData(args.year, lat, lon)
            solar_cf = getSolarCF(solar_data)
            col_name = f"{lat} {lon}"
            solar_df[col_name] = solar_cf

            # Wind
            wind_srw, iec_class = getWindSRW(args.year, lat, lon)
            wind_cf = getWindCF(wind_srw, iec_class, powerCurve, save_resource=args.save_resource)
            wind_df[col_name] = wind_cf

        except HTTPError as e:
            # If we hit a 429, let's save progress and bail out
            print(f"HTTPError {e.code} at (lat={lat}, lon={lon}).")
            if e.code == 429:
                print("Rate limit reached. Saving partial progress and exiting.")
                solar_df.to_csv(solar_filename)
                wind_df.to_csv(wind_filename)
                return
            else:
                print("Skipping this county due to HTTPError.")
                time.sleep(2)

        # Sleep to avoid hitting rate limits
        time.sleep(args.sleep_seconds)

        # Periodically save partial results
        if (i+1) % 50 == 0:
            solar_df.to_csv(solar_filename)
            wind_df.to_csv(wind_filename)
            if args.verbose:
                print("...saved partial progress...")

    # 5) Done. Save final
    solar_df.to_csv(solar_filename)
    wind_df.to_csv(wind_filename)

    end = datetime.datetime.now()
    print("Done!")
    print(f"Solar CF -> {solar_filename}")
    print(f"Wind  CF -> {wind_filename}")
    if args.verbose:
        print(f"Runtime: {end - start}")

if __name__ == "__main__":
    main()
