# %%
# standard lib
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
train_data = pd.read_csv(CFG.train_data_path)
test_data = pd.read_csv(CFG.test_data_path)
df = pd.concat([train_data, test_data], axis=0)
meas_col = [col for col in df.columns if col.startswith("measurement")]
print(train_data.head(5), train_data.shape)
# %%
for name, data, col in [
    (name, data, col)
    for name, data in {
        "train_data": train_data, "test_data": test_data
    }.items()
    for col in ("attribute_0", "attribute_1", "attribute_2", "attribute_3")
]:
    print(name, col, data[col].unique())
# %%
# product_code
fig = px.histogram(train_data, x="product_code", color="failure")
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.80)
fig.update_layout(
    margin_l=10,
    margin_b=10,
    margin_t=30,
    height=450,
)
if CREATE_PICTURE:
    fig.write_image(r"src\product_code_histogram.svg")
fig.show()
# %%
# attribute


def sunburst(data: pd.DataFrame, title: str):
    fig = px.sunburst(
        data,
        path=[
            "product_code",
            "attribute_3",
            "attribute_2",
            "attribute_1",
            "attribute_0",
        ]
    )
    fig.update_layout(
        title={
            "text": f"Sunburst {title}",
            "font": {"size": 22, "color": "black"},
            "x": 0.05,
            "y": 0.95,
        },
        margin_l=10,
        margin_b=10,
        margin_t=10,
        height=450,
    )
    if CREATE_PICTURE:
        fig.write_image(rf"src\attribute_sunburst_{title}.svg")
    fig.show()


sunburst(train_data, "train_data")
sunburst(test_data, "test_data")
# %%
# loading


def create_histogram(data: pd.DataFrame, x: str, color: str, title: str):
    fig = px.histogram(data, x=x, color=color)
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.80)
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 22, "color": "black"},
            "x": 0.5,
            "y": 0.95,
        },
        margin_l=10,
        margin_b=10,
        margin_t=30,
        height=450,
    )
    if CREATE_PICTURE:
        fig.write_image(rf"src\{title}_histogram.svg")
    fig.show()


# loading
create_histogram(train_data, "loading", "failure", "loading")
# log loading
log_train = train_data[["loading", "failure"]].copy()
log_train["log_loading"] = np.log(log_train["loading"])
create_histogram(log_train, "log_loading", "failure", "log_loading")
# area capacity
area = train_data["attribute_2"] * train_data["attribute_3"]
area_train = train_data[["loading", "failure"]].copy()
area_train["area_capacity"] = np.log(area_train["loading"] / area)
create_histogram(area_train, "area_capacity", "failure", "area_capacity")
# %%
# corr by product_code
for pcode in df["product_code"].unique():
    p_corr = df.query(f"product_code=='{pcode}'")[meas_col].corr("spearman")
    mask = np.tril(p_corr)
    p_corr = p_corr.where(np.abs(mask) > 0.05, 0)
    fig = px.imshow(p_corr)
    fig.update_layout(
        title={
            "text": f"correlation {pcode}",
            "font": {"size": 22, "color": "black"},
            "x": 0.5,
            "y": 0.95,
        },
        margin_l=5,
        margin_b=10,
        width=700,
        height=600,
    )
    if CREATE_PICTURE:
        fig.write_image(rf"src\correlation_{pcode}.svg")
    fig.show()
    for index, items in p_corr.iterrows():
        col = [col for col in items[items != 0.0].index if col != index]
        if len(col) == 0:
            continue
        sum_corr = np.abs(items[col]).sum()
        print(f"'{index}':'{pcode}':{col} ,{sum_corr}")
# %%
# measurement_3d
level_col = [col for col in meas_col if col.split("_")[-1] in ("0", "1", "2")]
fig = px.scatter_3d(
    df,
    x="measurement_0",
    y="measurement_1",
    z="measurement_2",
    color="product_code",
    opacity=0.7
)
fig.update_traces(
    marker={"size": 3}
)
fig.update_layout(
    margin_l=5,
    margin_b=10,
    width=700,
    height=600,
)
if CREATE_PICTURE:
    fig.write_image(r"src\measurement_3d.svg")
fig.show()
level_data = train_data[level_col + ["failure"]].copy()
level_data["level"] = level_data[level_col].apply(
    lambda x: np.log(np.prod(x + 1)), axis=1)
fig = px.histogram(level_data, x="level", color="failure")
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.80)
fig.update_layout(
    title={
        "text": "level prod",
        "font": {"size": 22, "color": "black"},
        "x": 0.5,
        "y": 0.95,
    },
    margin_l=10,
    margin_b=10,
    margin_t=30,
    height=450,
)
if CREATE_PICTURE:
    fig.write_image(r"src\level_histogram.svg")
fig.show()
# measurement
correlation_matrix = train_data[meas_col +
                                ["loading", "failure"]].corr("spearman")
# mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
fig = px.imshow(correlation_matrix)
fig.update_layout(
    margin_l=5,
    margin_b=10,
    width=700,
    height=600,
)
if CREATE_PICTURE:
    fig.write_image(r"src\measurement_corr.svg")
fig.show()
# %%
# missing
print(train_data.isnull().sum())
for col in ("measurement_0", "measurement_1", "measurement_2"):
    print(train_data[col].unique(), test_data[col].unique())
# %%
