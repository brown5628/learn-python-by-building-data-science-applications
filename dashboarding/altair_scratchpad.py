# %%
import pandas as pd
import altair as alt

# %%
alt.data_transformers.disable_max_rows()

# %%
data = pd.read_csv("../data/top5.csv", parse_dates=["date", ]).fillna(0)

# %%
len(data)

# %%
data.tail()

# %%
timeline = (
    alt.Chart(data, width=800)
    .mark_line()
    .encode(x=alt.X("date:T", timeUnit="yearmonthdate"), y="value", color="boro")
    .transform_filter((alt.datum.metric == "complaints"))
)

timeline

# %%
barchart = (
    alt.Chart(data, width=800)
    .mark_bar()
    .encode(
        x=alt.X("svalue:Q", title="Complaints Total"),
        y=alt.Y(
            "metric:N",
            sort=alt.EncodingSortField(
                field="svalue",  # The field to use for the sort
                order="descending",  # The order to sort in
            ),
        ),
        color=alt.value("purple"),
        tooltip=["metric", "svalue:Q"],
    )
    .transform_filter("datum.metric != 'complaints'")
    .transform_filter("datum.boro == 'NYC'")
    .transform_aggregate(svalue="sum(value)", groupby=["metric"])
    .transform_window(
        rank="rank(svalue)", sort=[alt.SortField("svalue", order="descending")]
    )
    .transform_filter("datum.rank <= 10")
)

barchart

# %%
brush = alt.selection_interval(encodings=["x"], empty="all")

T = timeline.add_selection(brush)

B = barchart.transform_filter(brush)

B.transform = [B.transform[-1], ] + B.transform[:-1]

dash = alt.vconcat(T, B, data=data)


# %%
dash

# %%
