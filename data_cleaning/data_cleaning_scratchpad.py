# %%
import pandas as pd

# %%
data = {"x": [1, 2, 3], "y": ["a", "b", "c"], "z": [False, True, False]}
df = pd.DataFrame(data)

# %%
df

# %%
df["y"]

# %%
df[["y", "z"]]

# %%
df.loc[[0, 1], "z"]

# %%
df["new_column"] = -1
df["new_column"]

# %%
df.iloc[-2:, 1:]

# %%
df[df["z"]]

# %%
mask = df["x"] == 2
mask

# %%
df.loc[mask, "y"]

# %%
N = pd.Series([1, 2, 3, 10])

# %%
N.mean()

# %%
N.median()

# %%
N.sum()

# %%
N.max()

# %%
df.shape

# %%
double = pd.concat([df, df], axis=0, sort=False)
double.shape

# %%
pd.concat([df, df], axis=1)

# %%
df.merge(df.head(2), on="y", how="left")

# %%
