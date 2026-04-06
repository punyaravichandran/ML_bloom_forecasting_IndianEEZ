"""
Spatial bloom forecasting in the Indian EEZ
Author: Punya P
"""

# Import the necessary python packages
import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from dask_ml.model_selection import GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor


# Earth Access Authentication
earthaccess.login(persist=True)


# Search Data
tspan = ("2024-01-01", "2024-06-30")

chl_results = earthaccess.search_data(
    short_name="MODISA_L3m_CHL",
    temporal=tspan,
    granule_name="*.DAY.*.4km.*"
)

sst_results = earthaccess.search_data(
    short_name="MODIS_AQUA_L3_SST_THERMAL_DAILY_4KM_DAYTIME_V2019.0",
    temporal=tspan,
    granule_name="*.DAY.*.4km.*"
)

par_results = earthaccess.search_data(
    short_name="MODISA_L3m_PAR",
    temporal=tspan,
    granule_name="*.DAY.*.4km.*"
)

flh_results = earthaccess.search_data(
    short_name="MODISA_L3m_FLH",
    temporal=tspan,
    granule_name="*.DAY.*.4km.*"
)


# Open Files
files_chl = earthaccess.open(chl_results)
files_sst = earthaccess.open(sst_results)
files_par = earthaccess.open(par_results)
files_flh = earthaccess.open(flh_results)


# Preprocess Function
def preprocess(ds):
    dt = ds.attrs["time_coverage_start"].replace("Z", "")
    ds["date"] = ((), np.datetime64(dt, "ns"))
    ds = ds.set_coords("date")
    return ds.sel(lat=slice(26, 2), lon=slice(64, 91))


# Load datasets
ds_chl = xr.open_mfdataset(files_chl, preprocess=preprocess,
                          combine="nested", concat_dim="date")

ds_sst = xr.open_mfdataset(files_sst, preprocess=preprocess,
                          combine="nested", concat_dim="date")

ds_par = xr.open_mfdataset(files_par, preprocess=preprocess,
                          combine="nested", concat_dim="date")

ds_flh = xr.open_mfdataset(files_flh, preprocess=preprocess,
                          combine="nested", concat_dim="date")


# Extract variables
chl = ds_chl["chlor_a"]
sst = ds_sst["sst"]
par = ds_par["par"]
flh = ds_flh["nflh"]


# Clip to Indian EEZ (shape file)
gdf = gpd.read_file("eez.shp")

if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326").dissolve()

def clip(da):
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(gdf.geometry)

chl = clip(chl)
sst = clip(sst)
par = clip(par)
flh = clip(flh)


# Align datasets
# remove duplicate indexes & check the spatial & temporal alignment 
chl = chl.sel(date=~chl.indexes['date'].duplicated())
sst = sst.sel(date=~sst.indexes['date'].duplicated())
par = par.sel(date=~par.indexes['date'].duplicated())
flh = flh.sel(date=~flh.indexes['date'].duplicated())

sst = sst.reindex(date=chl.date, method="nearest")
par = par.reindex(date=chl.date, method="nearest")
flh = flh.reindex(date=chl.date, method="nearest")

sst = sst.reindex(lat=chl.lat, lon=chl.lon, method="nearest")
par = par.reindex(lat=chl.lat, lon=chl.lon, method="nearest")
flh = flh.reindex(lat=chl.lat, lon=chl.lon, method="nearest")

combined = xr.merge([chl, sst, par, flh], compat="override")


# Log transform
combined["chlor_a"] = np.log10(
    combined["chlor_a"].where(combined["chlor_a"] > 0.1)
)

# Stack for ML
stacked = combined.stack(samples=("date", "lat", "lon")).dropna("samples")

X = xr.concat(
    [stacked.sst, stacked.par, stacked.nflh],
    dim="feature"
).transpose("samples", "feature").data

y = stacked.chlor_a.data

X_np = X.compute()
y_np = y.compute()


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)


# Machine Learning Models
# 1. Random Forest + GridSearch
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
}

grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Best Parameters for RF:", grid.best_params_)

# 2. SVM + GridSearch
param_grid = {
    "C": [0.1, 1, 10],
    "epsilon": [0.1, 0.2, 0.5],
}

grid = GridSearchCV(SVR(), param_grid, cv=3)
grid.fit(X_train, y_train)

best = grid.best_estimator_
print("Best Parameters for SVM:", grid.best_params_)

#3. XGBoost + GridSearch
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
}

grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)

best = grid.best_estimator_

print("Best Parameters for XGBoost:", grid.best_params_)

# Full model prediction
y_pred_full = model.predict(X_np)

# Test data prediction
y_pred_test = model.predict(X_test)

r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)


# 15. Scatter Plot
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Chlorophyll Prediction")
plt.grid()

plt.show()

#Partial Dependence Plot
from sklearn.inspection import PartialDependenceDisplay
cleaned_stacked_data_pandas = stacked.to_pandas()
feature_names = ['sst', 'par','nflh']
PartialDependenceDisplay.from_estimator(
    best, 
    cleaned_stacked_data_pandas[['sst', 'par','nflh']],
    feature_names,
    kind='average' # or 'individual'
)
plt.suptitle("Partial Dependence Plots: SST and PAR vs. Predicted Chlorophyll")
plt.show()

# Model Bias
# Rebuild MultiIndex & Unstack
samples = stacked.samples.values
dates, lats, lons = zip(*samples)

multi_index = pd.MultiIndex.from_arrays(
    [dates, lats, lons],
    names=["date", "lat", "lon"]
)

# Predicted array
pred_da = xr.DataArray(
    y_pred_full,
    coords={"samples": multi_index},
    dims=["samples"]
)
pred_unstack = pred_da.unstack()

# Actual array
actual_da = xr.DataArray(
    y_np,
    coords={"samples": multi_index},
    dims=["samples"]
)
actual_unstack = actual_da.unstack()

# Calculate mean over date dimension
mean_actual_chl = actual_unstack.mean("date")
mean_predicted_chl = pred_unstack.mean("date")
difference_chl = mean_predicted_chl - mean_actual_chl

# Plot Bias
plt.figure(figsize=(8, 6))
plot_diff = difference_chl.plot.imshow(
    cmap="RdBu",
    center=0,
    cbar_kwargs={'orientation': 'vertical', 'pad': 0.02}
)
plt.title("Difference (Predicted - Actual) on Log10 Scale", fontsize=16)
plt.xlabel("Longitude", fontsize=17)
plt.ylabel("Latitude", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plot_diff.colorbar.ax.tick_params(labelsize=15)
plt.tight_layout()
plt.show()
