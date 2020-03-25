# %%
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# %%
data = pd.read_csv("../data/EF_battles_corrected.csv", parse_dates=["start", "end"])
data.shape

# %%
data["end_num"] = (data["end"].dt.year - 1938) * 12 + data["end"].dt.month
data["start_num"] = (data["start"].dt.year - 1938) * 12 + data["start"].dt.month
data["duration"] = (data["end_num"] - data["start_num"]).clip(lower=1)

# %%
data.shape

# %%
data.dtypes

# %%
cols = [
    "allies_infantry",
    "axis_infantry",
    "allies_tanks",
    "axis_tanks",
    "allies_guns",
    "axis_guns",
]

# %%
mask = data[["allies_infantry", "axis_infantry"]].notnull().all(1)
data_kmeans = data.loc[mask, cols].fillna(0)

# %%
data_kmeans.shape

# %%
data["result_num"] = data["result"].map({"axis": -1, "allies": 1}).fillna(0)

# %%
data["result_num"].value_counts()

# %%
model = KMeans(n_clusters=5, random_state=2019)


# %%
labels = model.fit_predict(data_kmeans) + 1

# %%
print(labels)

# %%
data_kmeans["label"] = ("Cluster " + pd.Series((labels)).astype(str)).values
data_kmeans[["name", "result", "start"]] = data.loc[mask, ["name", "result", "start"]]

# %%
c = (
    alt.Chart(data_kmeans)
    .mark_point()
    .encode(
        shape=alt.Shape("label:N", legend=alt.Legend(title="Cluster")),
        x="allies_infantry",
        y="axis_infantry",
        color="result",
        tooltip=data_kmeans.columns.tolist(),
    )
    .interactive()
)

c

# %%
data_to_scale = data_kmeans.drop(["label", "name", "start", "result"], axis=1)
data_scaled = scale(data_to_scale)

# %%
labels_scaled = model.fit_predict(data_scaled) + 1

# %%
data_kmeans["label 2"] = ("Cluster" + pd.Series((labels_scaled)).astype(str)).values

# %%
c.data = data_kmeans

# %%
c.encode(shape=alt.Shape("label 2:N", legend=alt.Legend(title="Cluster")))

# %%
centroids = pd.DataFrame(model.cluster_centers_, columns=data_to_scale.columns)
centroids.index += 1

# %%
centroids

# %%
