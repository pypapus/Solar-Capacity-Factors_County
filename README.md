# Eastern Counties Solar & Wind Capacity Factor Download

This project contains a Python script, **`eastern_counties.py`**, which downloads hourly solar and wind resource data for **all counties in the Eastern Interconnect** (approximated by full states). It then uses **PySAM** to compute hourly capacity factors for each county. Two CSV files (one for solar, one for wind) are saved to an output folder.

## Table of Contents
1. [Features](#features)
2. [Dependencies](#dependencies)
3. [Folder Structure](#folder-structure)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Notes & Tips](#notes--tips)
7. [License](#license)

---

## Features

- **County-level** resolution: Computes one or more representative points (centroids) per county.
- **Approximate Eastern Interconnect**: Filters counties by state. (You can refine the script if you need more precise boundaries.)
- **Hourly capacity factors**:  
  - *Solar:* Uses NREL’s NSRDB PSM3 data and PySAM’s PVWatts v8 model.  
  - *Wind:* Uses NREL’s WIND Toolkit data and PySAM’s Windpower model.
- **Handles partial downloads**: If the script is interrupted, you can re-run it to continue where it left off.
- **Adjustable rate-limiting**: You can specify a `--sleep_seconds` delay between requests to avoid exceeding NREL’s API limits.

---

## Dependencies

- **Python 3.7+** recommended
- **PySAM**  
- **geopandas**  
- **shapely**, **fiona**, **rtree** (installed automatically with geopandas in most distributions)  
- **pandas**, **numpy**  
- Other standard Python libraries: `time`, `datetime`, `argparse`, etc.

### Installation Example

If you’re using **pip**:
```bash
pip install NREL-PySAM geopandas shapely fiona rtree pandas numpy
```

If you’re using **conda** (Anaconda/Miniconda):
```bash
conda create -n myenv python=3.9
conda activate myenv
conda install -c conda-forge geopandas shapely fiona rtree
pip install NREL-PySAM
```
(Adjust Python version and environment name as desired.)

---

## Folder Structure

A possible layout for your project:

```
eastern_interconnect_project/
│
├── eastern_counties.py        # The main script
├── powerCurves/
│   └── powerCurves.csv        # Composite wind power curves by IEC class
├── resourceData/              # (auto-created) Storage for downloaded .csv (solar) and .srw (wind) files
├── output/                    # (auto-created) Final CSVs of capacity factors
├── README.md                  # This README
└── environment.yml            # (optional) If using conda environment
```

---

## Usage

1. **Download or clone this repository** (or place `eastern_counties.py` in a folder of your choice).

2. **Acquire a nationwide counties shapefile** (e.g., from [US Census TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)).  
   - Recommended: *Counties (1:5,000,000)* or another nationwide counties layer that has **`STUSPS`** columns with the 2-letter state abbreviations.

3. **Obtain an NREL API key** by [signing up here](https://developer.nrel.gov/signup/).

4. **Run the script**:
   ```bash
   python eastern_counties.py \
       --year 2012 \
       --api-key <YOUR_NREL_API_KEY> \
       --email <YOUR_EMAIL> \
       --counties_shapefile "C:/Path/To/Counties.shp" \
       --save_resource \
       --verbose \
       --sleep_seconds 2
   ```

   - `--year`: One of `2007` to `2014`.  
   - `--api-key`: Your NREL API key.  
   - `--email`: The email associated with your NREL account.  
   - `--counties_shapefile`: A **nationwide** counties shapefile path (any format geopandas can read).  
   - `--save_resource`: Keep the downloaded NSRDB `.csv` and WIND Toolkit `.srw` files (otherwise they’re deleted after processing).  
   - `--verbose`: Prints extra info to the console.  
   - `--sleep_seconds`: Delay in seconds between each county download request to avoid HTTP 429 (Too Many Requests). Defaults to 1. Increase if you hit rate limits.

5. **Check the output**:
   - By default, **two CSV files** will appear in the `output` folder:  
     - `eastern_county_solar_cf_<year>.csv`  
     - `eastern_county_wind_cf_<year>.csv`
   - Each CSV’s columns are labeled `lat lon` for each county. Rows correspond to **hourly** data for that year.

---

## How It Works

1. **Loads** the nationwide county shapefile via `geopandas`.  
2. **Filters** to states in `EASTERN_INTERCONNECT_STATES`.  
3. **Computes the centroid** of each county polygon.  
4. **Requests data** from the NREL APIs:  
   - **NSRDB** (for solar) → downloads a .csv file with DNI, GHI, DHI, wind speed, etc.  
   - **WIND Toolkit** (for wind) → downloads a `.srw` file containing hourly wind speed/direction.  
5. **Runs PySAM** (PVWatts for solar, Windpower for wind) to turn resource data into hourly capacity factors.  
6. **Stores** the capacity factor time series in CSV format, with each column representing one county.

---

## Notes & Tips

1. **Partial Restarts**  
   - If you kill the script or encounter an error, the next run **automatically skips** counties that already have data in the CSVs.

2. **Rate Limit**  
   - NREL’s free tier may return **HTTP 429** if you exceed the request limit. The script will **save progress** and stop. After waiting some time (or increasing `--sleep_seconds`), you can rerun it.

3. **Wind Data**  
   - The default script uses a **1.5 MW** single-turbine farm with default `powerCurves.csv` to compute wind power output.

4. **Refining the Eastern Interconnect**  
   - This script simply uses entire states. However, some states are partially in other interconnections (like Texas, Oklahoma, Kansas). If you need precise boundaries, **edit the shapefile** or filter out counties not in the Eastern Interconnect boundary.

5. **Bigger Datasets**  
   - If you’re downloading data for many counties, it can take a while. Consider using a more powerful system or a HPC environment. Increase the sleep or request a higher API quota from NREL.

---

