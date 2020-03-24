# %%
import pandas as pd
import pylab as plt
import geopandas as gp

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
