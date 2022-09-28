# %%
# standard lib
from dataclasses import dataclass
from pathlib import Path
# third party
import numpy as np
import pandas as pd
from plotly import express as px

CREATE_IMAGE = False


@dataclass
class Config:
    train_data_path: Path = Path(r"Data\train.csv")
    test_data_path: Path = Path(r"Data\test.csv")
    sub_data_path: Path = Path(r"Data\sample_submission.csv")


CFG = Config()
# %%
train_data = pd.read_csv(CFG.train_data_path).set_index("row_id")
test_data = pd.read_csv(CFG.test_data_path).set_index("row_id")
print(train_data.head(5), train_data.shape, sep="\n")
print(
    train_data["country"].unique(),
    train_data["store"].unique(),
    train_data["product"].unique(),
    sep="\n"
)
# %%
# location
trans_isocode = {
    'Belgium': "BEL",
    'France': "FRA",
    'Germany': "DEU",
    'Italy': "ITA",
    'Poland': "POL",
    'Spain': "ESP"
}
fig = px.choropleth(
    train_data.replace(trans_isocode),
    color="num_sold",
    locations="country",
    scope="europe",
    animation_frame="date",
    range_color=[0, 1000]
)
fig.update_geos(
    # fitbounds="locations",
    visible=True,
    center={"lat": 48.0, "lon": 7.0},
    projection_scale=2.8,
    showland=True,
    showocean=True,
    landcolor="LightGreen",
    oceancolor="LightBlue",
)
fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=700
)
fig.show()
if CREATE_IMAGE:
    fig = px.choropleth(
        (
            train_data
            .groupby("country", as_index=False)
            .median()
            .replace(trans_isocode)
        ),
        color="num_sold",
        locations="country",
        scope="europe",
        range_color=[0, 1000]
    )
    fig.update_geos(
        # fitbounds="locations",
        visible=True,
        center={"lat": 48.0, "lon": 7.0},
        projection_scale=2.8,
        showland=True,
        showocean=True,
        landcolor="LightGreen",
        oceancolor="LightBlue",
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700
    )
    fig.write_image(r"src\location_numsold.svg")

# %%
# time scale


def timescale_lineplot(data: pd.DataFrame, title: str):
    fig = px.line(
        data,
        x="date",
        y="num_sold",
        color="product",
        facet_row="store"
    )
    fig.update_layout(
        margin={"r": 10, "t": 10, "l": 10, "b": 40},
        height=600,
        legend={
            "yanchor": "top",
            "y": 0.48,
            "xanchor": "left",
            "x": 0.01
        },
        title={
            "text": title,
            "font": {"size": 22, "color": "black"},
            "x": 0.95,
            "y": 0.99,
        },
        xaxis={
            "rangeslider": {"visible": True, "thickness": 0.01},
            "type": "date"
        },
        yaxis_range=[0, 1000]
    )
    if CREATE_IMAGE:
        fig.write_image(rf"src\timescale_{title}.svg")
    fig.show()


for country in train_data["country"].unique():
    timescale_lineplot(
        train_data.query(f"country=='{country}'"),
        country
    )
# %%
# sunburst
fig = px.sunburst(
    train_data,
    path=[
        "country",
        "store",
        "product"
    ],
    values="num_sold"
)
fig.update_layout(
    title={
        "text": "Sunburst num_sold",
        "font": {"size": 22, "color": "black"},
        "x": 0.05,
        "y": 0.95,
    },
    margin_l=10,
    margin_b=10,
    margin_t=10,
    height=450,
)
if CREATE_IMAGE:
    fig.write_image(r"src\numsold_sunburst.svg")
fig.show()
# %%


def auto_corr(df: pd.Series, k: int) -> pd.DataFrame:
    y_avg = df.mean()
    df_len = len(df)
    data = df.to_list()

    sum_of_covariance = 0
    for i in range(k, df_len):
        covar = (data[i] - y_avg) * (data[i - k] - y_avg)
        sum_of_covariance += covar

    sum_of_denominator = 0
    for j in range(df_len):
        demoni = np.square(data[j] - y_avg)
        sum_of_denominator += demoni

    return sum_of_covariance / sum_of_denominator


def plot_bar(df: pd.DataFrame, title: str):
    fig = px.bar(
        df,
        x="lag",
        y="auto_corr",
        color="store",
        barmode="group",
    )
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 22, "color": "black"},
            "x": 0.1,
            "y": 0.95,
        },
        legend={
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "right",
            "x": 0.99
        },
        margin={"r": 10, "t": 10, "l": 0, "b": 0},
        height=450,
    )
    if CREATE_IMAGE:
        fig.write_image(rf"src\auto_corr_{title}.svg")
    fig.show()


LAG_RANGE = 366
store_list = train_data["store"].unique()
for country in train_data["country"].unique():
    res_dict = {}
    for store in store_list:
        data = train_data.query(f"country=='{country}' and store=='{store}'")
        data = data[["date", "num_sold"]].groupby("date").mean()
        res_dict[store] = [
            auto_corr(train_data["num_sold"], count)
            for count in range(LAG_RANGE)
        ]
    data = (
        pd
        .DataFrame(res_dict, index=range(LAG_RANGE))
        .stack(level=0)
        .reset_index()
        .rename(columns={
            "level_0": "lag",
            "level_1": "store",
            0: "auto_corr"
        })
    )
    plot_bar(data, country)
# %%
product_ratio_df = (
    train_data
    .groupby(["date", "product"])["num_sold"]
    .sum()
    .reset_index()
    .pivot(index="date", columns="product", values="num_sold")
    .apply(lambda x: x / x.sum(), axis=1)
    .stack()
    .rename("ratios")
    .reset_index()
)
product_ratio_df.head()
fig = px.line(
    product_ratio_df,
    x="date",
    y="ratios",
    color="product",
)
fig.update_layout(
    margin={"r": 10, "t": 10, "l": 10, "b": 40},
    height=600,
    legend={
        "orientation": "h",
        "yanchor": "top",
        "y": 0.99,
        "xanchor": "left",
        "x": 0.1
    },
    xaxis={
        "rangeslider": {"visible": True, "thickness": 0.01},
        "type": "date"
    },
    yaxis_range=[0.14, 0.39]
)
if CREATE_IMAGE:
    fig.write_image(r"src\seasonally.svg")
fig.show()
# %%
