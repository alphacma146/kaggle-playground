# %%
# standard lib
from dataclasses import dataclass
from pathlib import Path
# third party
import numpy as np
import pandas as pd
from plotly import express as px

CREATE_IMAGE = True


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
        width=800,
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
