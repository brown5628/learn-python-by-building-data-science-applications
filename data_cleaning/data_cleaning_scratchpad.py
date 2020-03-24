# %%
import pandas as pd
import json
import missingno as msno
# import pylab as plt
from copy import copy
from pandas.io.json import json_normalize
from geocode import nominatim_geocode
from tqdm import tqdm

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
path = "../web_scraping/all_battles_parsed.json"
with open(path, "r") as f:
    battles = json.load(f)


# %%


def _flatten_battles(battles, root=None):
    battles_to_run = copy(battles)
    records = []

    for name, data in battles.items():
        if "children" in data:
            children = data.pop("children")
            records.extend(_flatten_battles(children, root=name))
        else:
            data["name"] = name
            data["parent"] = root
            records.append(data)

    return records


# %%
records = {k: _flatten_battles(v, root=k) for k, v in battles.items()}

# %%
records = {k: pd.DataFrame(json_normalize(v)) for k, v in records.items()}

# %%
for front, data in records.items():
    data.to_csv(f"../data/{front}.csv", index=None)

# %%
data = pd.read_csv("../data/Eastern Front.csv")
data.shape

# %%
data.dtypes

# %%
data.level.value_counts()

# %%
battles = data[data.level == 1]
battles.shape

# %%
columns = [
    "Location",
    "name",
    "Date",
    "Result",
    "Belligerents.allies",
    "Belligerents.axis",
    "Casualties and losses.allies",
    "Casualties and losses.axis",
]

# %%
battles[columns].head(3)

# %%
msno.matrix(battles, labels=True, sparkline=False)

# %%
mask = battles[["Date", "Location"]].isnull().all(1)

# %%
print(battles.loc[mask, ["name", "url"]].to_string())

# %%
battles = battles.dropna(subset=["Date", "Location"])

# %%
pattern = r"/ ([\d|\.]+); ([\d|\.]+)"

# %%
battles.head(10).Location.str.extract(pattern)

# %%
battles[["Latitude", "Longitude"]] = battles.Location.str.extract(pattern)

# %%
for col in "Latitude", "Longitude":
    battles[col] = battles[col].astype(float)

# %%
f"{(battles['Latitude'].isnull().sum() / len(battles)):.1%}"

# %%
tqdm().pandas()

# %%
geo_mask = battles["Latitude"].isnull()
battles.loc[geo_mask, "Location"].sample(15, random_state=2019)

# %%
location = battles["Location"].str.lower().str.replace("near ", "")

# %%
replacements = {
    "Ukrainian SSR, Soviet Union": "Ukraine",
    "Russian SFSR, Soviet Union": "Russia",
    "Russian SFSR": "Russia",
    "Belorussian SSR": "Belorus",
    "Soviet Union": "",
    "USSR": "",
    ", Poland (now Ukraine)": "Ukraine",
    "east prussia (now kaliningrad oblast)": "Kaliningrad Oblast, Russia",
    ", czechoslovakia": ", czech republic",
    "königsberg, germany (now: kaliningrad, russia)": "Kaliningrad Oblast, Russia",
    "lwów, lwów voivodeship, poland": "Lvov, Ukraine",
    "leningrad region, ; narva, estonia": "Narva, Estonia",
    "Kingdom of Hungary": "Hungary",
    "odessa region, ukraine": "Odessa, Ukraine",
}

# %%
for k, v in replacements.items():
    location = location.str.replace(k.lower(), v.lower(), regex=False)

# %%


def vectorized_geocode(x):
    result = nominatim_geocode(x)
    if len(result) == 0:
        return dict()
    return {k: result[0][k] for k in ("lat", "lon", "importance", "display_name")}


# %%
location[location.fillna("").str.contains("kreisgebiete")] = "Courland Peninsula"


# %%
response = location[geo_mask].str.replace("\n", " ").progress_apply(vectorized_geocode)

# %%
response.iloc[0]

# %%
battles.loc[geo_mask, "Location"].iloc[0]

# %%
geo_df = pd.DataFrame(response.tolist(), index=response.index)
geo_df.rename(columns={"lat": "Latitude", "lon": "Longitude"}, inplace=True)

# %%
rmask = geo_df["importance"].isnull()

# %%
f"{rmask.sum() / len(battles):.1%}"

# %%
location[geo_mask].loc[mask]

# %%
d = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    " August",
    "September",
    "October",
    "November",
    "December",
)

month_pattern = r"(" + "|".join(d) + ")"
year_pattern = r"(19\d\d)"
# %%
year_extracted = battles["Date"].str.extractall(year_pattern).unstack()

# %%
year_extracted[year_extracted.iloc[:, -1].notnull()]

# %%
year_extracted = year_extracted.iloc[:, :2]

# %%
year_extracted.head(10)

# %%
year_extracted.iloc[:, 1] = year_extracted.iloc[:, 1].fillna(year_extracted.iloc[:, 0])

# %%
month_extracted = battles["Date"].str.extractall(month_pattern).unstack()

# %%
for i in range(2, month_extracted.shape[1] + 1):
    month_extracted.iloc[:, -1].fillna(month_extracted.iloc[:, -i], inplace=True)

# %%
month_extracted = month_extracted.iloc[:, [0, -1]]

# %%
year_extracted.columns = month_extracted.columns = ["start", "end"]
I = battles.index
cols = "start", "end"

for col in cols:
    battles[col] = pd.to_datetime(
        month_extracted.loc[I, col] + " " + year_extracted.loc[I, col]
    )

# %%
words = ["Germany", "Italy", "Estonian conscripts"]

# %%
for word in words:
    mask = battles["Belligerents.allies"].fillna("").str.contains(word)
    axis_party = battles.loc[
        mask, ["Belligerents.allies", "Casualties and losses.allies"]
    ].copy()
    battles.loc[
        mask, ["Belligerents.allies", "Casualties and losses.allies"]
    ] = battles.loc[mask, ["Belligerents.axis", "Casualties and losses.axis"]].values
    battles.loc[
        mask, ["Belligerents.axis", "Casualties and losses.axis"]
    ] = axis_party.values

# %%
battles["Casualties and losses.allies"].iloc[0]

# %%
digit_pattern = "([\d|\,|\.]+)(?:\[\d+\])?\+?\s*(?:{words})"

keywords = {
    "killed": ["men", "dead", "killed", "casualties", "kia"],
    "wounded": ["wounded", "sick", "injured"],
    "captured": ["captured", "prisoners"],
    "tanks": ["tank", "panzer"],
    "airplane": ["airplane", "aircraft"],
    "guns": ["artillery", "gun", "self propelled guns", "field-guns", "anti-tank guns"],
    "ships": ["warships", "boats", "destroyer", "minelayer"],
    "submarines": ["submarines"],
}

only_digits = "([\d|\,|\.]+)\Z"

# %%
def _shy_convert_numeric(v):
    if pd.isnull(v) or v in (",", "."):
        return pd.np.nan

    return int(v.replace(",", "").replace(".", ""))


# %%
results = {
    "allies": pd.DataFrame(
        index=battles.index, columns=keywords.keys()
    ),  # empty dataframes with the same index
    "axis": pd.DataFrame(index=battles.index, columns=keywords.keys()),
}

for name, edf in results.items():
    column = battles[f"Casualties and losses.{name}"]
    for tp, keys in keywords.items():
        pattern = digit_pattern.format(words="|".join(keys))
        print(pattern)
        extracted = column.str.lower().str.extractall(pattern).unstack()
        values = extracted.applymap(_shy_convert_numeric)
        #         if tp == 'killed':
        #             mask values.iloc[:, 0].notnull()
        edf[tp] = values.min(1)
    results[name] = edf.fillna(0).astype(int)

    b = (
        column.fillna("")
        .str.extract(only_digits)
        .applymap(_shy_convert_numeric)
        .iloc[:, 0]
    )
    mask = b.notnull()
    results[name].loc[mask, "killed"] = b[mask]

# %%
pattern = "([\d|\,]+)(?:\[\d+\])?\+?\s*(?:men|dead|killed|casualties|kia)"
battles[f"Casualties and losses.axis"].str.extractall(pattern).unstack().head(5)

# %%
battles["Casualties and losses.axis"][battles.name == "Battle of Stalingrad"].iloc[0]

# %%
results["axis"][battles.name == "Battle of Stalingrad"]

# %%
battles.loc[[27], "Casualties and losses.axis"].iloc[0]

# %%
results["axis"].loc[27]

# %%
results["old_metrics"] = battles
new_dataset = pd.concat(results, axis=1)

# %%
idx = pd.IndexSlice

# %%
assumptions = {
    "killed": [0, 2_000_000],
    "wounded": [0, 1_000_000],
    "tanks": [0, 5_000],
    "airplane": [0, 3_000],
    "guns": [0, 30_000],
    ("start", "end"): [pd.to_datetime(el) for el in ("1939-01-01", "1945-12-31")],
}

# %%
def _check_assumptions(data, assumptions):
    for k, (min_, max_) in assumptions.items():
        df = data.loc[:, idx[:, k]]
        for i in range(df.shape[1]):
            assert df.iloc[:, i].between(min_, max_).all(), (
                df.iloc[:, i].name,
                df.iloc[:, i].describe(),
            )


# %%
d = new_dataset.loc[
    new_dataset.loc[:, idx["allies", "tanks"]] > 1_000,
    idx["old_metrics", ["name", "url"]],
]
d

# %%
d.iloc[0, 1]

# %%
_check_assumptions(new_dataset, assumptions)

# %%
new_dataset.loc[
    ~new_dataset.loc[:, idx["old_metrics", "start"]].between(
        *[pd.to_datetime(el) for el in ("1939-01-01", "1945-12-31")]
    ),
    idx["old_metrics", ["start"]],
]

# %%
new_dataset.loc[
    new_dataset.loc[:, idx["old_metrics", "start"]].isnull(),
    idx["old_metrics", ["name", "url"]],
]

# %%
new_dataset.loc[135, idx["old_metrics", "start"]] = pd.to_datetime("1944-08-09")
new_dataset.loc[135, idx["old_metrics", "end"]] = pd.to_datetime("1944-08-16")

# %%
_check_assumptions(new_dataset, assumptions)

# %%
new_dataset.to_csv("../data/EF_battles.csv", index=None)

# %%
