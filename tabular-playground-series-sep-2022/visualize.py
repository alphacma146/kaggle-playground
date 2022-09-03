# %%
# standard lib

import json
from dataclasses import dataclass
from pathlib import Path
# third party
import numpy as np
import pandas as pd
from plotly import express as px

CREATE_PICTURE = False


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
df = train_data.groupby("country", as_index=False).aggregate("sum")
df["iso_code"] = ["BEL", "FRA", "DEU", "ITA", "POL", "ESP"]
fig = px.choropleth(
    df,
    color="num_sold",
    locations="iso_code",
    scope="europe",
)
fig.update_geos(
    # fitbounds="locations",
    visible=True,
    center={"lat": 48.0, "lon": 7.0},
    projection_scale=2.8
)
fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=700,
)
fig.show()

# %%
