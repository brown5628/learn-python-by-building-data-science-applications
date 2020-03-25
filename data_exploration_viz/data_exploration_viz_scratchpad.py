# %%
import pandas as pd
import pylab as plt
import geopandas as gp
import numpy as np

# %%
raw_data = pd.read_csv("../data/EF_battles_corrected.csv", parse_dates=["start", "end"])

# %%
cols = [
    "name",
    "allies killed",
    "axis killed",
    "allies_tanks",
    "axis_tanks",
    "allies_planes",
    "axis_planes",
    "latlon",
    "start",
    "end",
    "url",
    "parent",
]

data = raw_data[cols].set_index("name")

# %%
data.head(3)

# %%
kill_cols = ["allies killed", "axis killed"]
data["killed total"] = data[kill_cols].sum(1)
data["killed total"].sort_values(ascending=False).head(3)

# %%
mask = data[kill_cols].isnull().any(1) | (data[kill_cols] == 0).any(1)

# %%
data.loc[~mask, "killed total"].median()

# %%
data.loc[~mask, ["allies_tanks", "axis_tanks"]].describe()

# %%
data.loc[~mask, "killed total"].hist(bins=20, figsize=(10, 10))
plt.suptitle("Histogram, overall casualties per battle")
plt.xlabel("killed")
plt.ylabel("frequency")
plt.tight_layout()

# %%
aggr = (
    data[~mask]
    .groupby("parent")
    .agg(
        {
            "axis killed": ["sum", "median", "count"],
            "allies killed": ["sum", "median"],
            "killed total": ["sum", "median"],
        }
    )
    .astype(int)
)
aggr
# %%
idx = pd.IndexSlice
aggr[idx["axis killed", "sum"]].head(3)

# %%
aggr.loc[:, idx[:, "sum"]].head(3)

# %%
aggr.plot(
    kind="scatter",
    x=idx["allies killed", "sum"],
    y=idx["axis killed", "sum"],
    figsize=(7, 7),
    title="Deaths on both sides",
)

plt.axis("equal")
plt.tight_layout()

# %%
ts = data[["axis killed", "allies killed", "end"]].copy()
ts = ts.set_index("end").sort_index()
r = ts.resample("1Y").agg("sum")
r

# %%
r.plot()

# %%
url = "https://unpkg.com/world-atlas@1/world/50m.json"
MERCATOR = {"init": "epsg:4326", "no_defs": True}
borders = gp.read_file(url)
borders.crs = MERCATOR

# %%
borders.plot(figsize=(10, 5))

# %%
borders = borders.to_crs(epsg=3035)

# %%
data_split = data.copy()
data_split.head()

# %%
Latitude = []
Longitude = []

for row in data_split["latlon"]:
    try:
        Latitude.append(row.split(",")[0])
        Longitude.append(row.split(",")[1])
    except:
        Latitude.append(np.NaN)
        Longitude.append(np.NaN)

data_split["Latitude"] = pd.to_numeric(Latitude)
data_split["Longitude"] = pd.to_numeric(Longitude)


data_split.head()

# %%
data = data_split

# %%
gdf = gp.GeoDataFrame(
    data, geometry=gp.points_from_xy(data["Longitude"], data["Latitude"]), crs=MERCATOR
).to_crs(borders.crs)

# %%
ax = borders.plot(color="lightgrey", edgecolor="white", figsize=(12, 12))
gdf.plot(
    ax=ax,
    color="red",
    markersize=(data["killed total"] / 1000).clip(lower=1),
    alpha=0.2,
)

ax.margins(x=-0.4, y=-0.4)
ax.set_axis_off()
