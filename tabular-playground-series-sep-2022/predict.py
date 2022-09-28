# %%
# Standard lib
from pathlib import Path
from dataclasses import dataclass, field
import datetime
# Third party
import plotly.express as px
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import holidays

PARAM_SEARCH = True


@dataclass
class Config:
    train_path: Path = Path(r"Data\train.csv")
    test_path: Path = Path(r"Data\test.csv")
    sample_path: Path = Path(r"Data\sample_submission.csv")
    result_folder: Path = Path(r"Data")
    model_params: dict = field(default_factory=lambda: {
        "device": "gpu",
        "objective": "regression",
        "boosting_type": "gbdt",
        "max_bins": 255,
        "metric": "rmse",
        "learning_rate": 0.05
    })
    national_holidays: dict = field(default_factory=lambda: {
        'Belgium': holidays.BE,
        'France': holidays.FR,
        'Germany': holidays.DE,
        'Italy': holidays.IT,
        'Poland': holidays.PL,
        'Spain': holidays.ES
    })
    # 一人当たりのGDP（USドル）
    national_GDP: dict = field(default_factory=lambda: {
        'Belgium': {
            2017: 44_274,
            2018: 47_683,
            2019: 46_733,
            2020: 45_255,
            2021: 51_875,
        },
        'France': {
            2017: 40_054,
            2018: 43_021,
            2019: 41_939,
            2020: 40_162,
            2021: 44_853,
        },
        'Germany': {
            2017: 44_637,
            2018: 47_995,
            2019: 46_800,
            2020: 46_216,
            2021: 50_795,
        },
        'Italy': {
            2017: 32_649,
            2018: 34_918,
            2019: 33_628,
            2020: 31_707,
            2021: 35_473,
        },
        'Poland': {
            2017: 13_869,
            2018: 15_468,
            2019: 15_727,
            2020: 15_718,
            2021: 17_815,
        },
        'Spain': {
            2017: 28_197,
            2018: 30_423,
            2019: 29_576,
            2020: 27_039,
            2021: 30_090,
        }
    })


CFG = Config()
# %%


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    preprocess
    """
    data["date"] = pd.to_datetime(data["date"])
    for factor in (
        "day",
        "dayofyear",
        "dayofweek",
        "month",
        "quarter",
        "year"
    ):
        data[factor] = getattr(data["date"].dt, factor)
    data["weekend"] = data["dayofweek"] >= 5
    data["holiday"] = data[["date", "year", "country"]].apply(
        lambda x: int(
            datetime.date(x[0].year, x[0].month, x[0].day)
            in CFG.national_holidays[x[2]](years=x[1]).keys()
        ),
        axis=1
    )
    data["GDP"] = data[["year", "country"]].apply(
        lambda x: CFG.national_GDP[x[1]][x[0]],
        axis=1
    )
    data.set_index("row_id", inplace=True)

    data.drop(["date", "country", "store", "product"], axis=1, inplace=True)

    return data


train_data = preprocess(pd.read_csv(CFG.train_path))
test_data = preprocess(pd.read_csv(CFG.test_path))
target_value = train_data["num_sold"].copy()
train_data.drop(["num_sold"], axis=1, inplace=True)
print("train_data:", train_data.shape)
print("test_data: ", test_data.shape)
# %%
# lgb parameter tuning
match PARAM_SEARCH:
    case True:
        train_set = lgb.Dataset(train_data, target_value)
        tuner = opt_lgb.LightGBMTunerCV(
            CFG.model_params,
            train_set=train_set,
            num_boost_round=500,
            shuffle=True,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ]
        )
        tuner.run()
        params = tuner.best_params
    case False:
        params = {
            'feature_pre_filter': False,
            'lambda_l1': 1.5962295482331676,
            'lambda_l2': 6.51631379485157e-07,
            'num_leaves': 8,
            'feature_fraction': 0.82,
            'bagging_fraction': 0.94782375381206,
            'bagging_freq': 4,
            'min_child_samples': 25
        }
        # roc_auc Score: 0.9237822837583836
# %%
# lightgbm model
print(params)
lgb_model = lgb.LGBMClassifier(**params, num_iterations=500)
lgb_model.fit(
    train_data,
    target_value,
    callbacks=[
        # lgb.early_stopping(100),
        lgb.log_evaluation(10),
    ]
)
importance = pd.DataFrame(
    data={
        "col": train_data.columns,
        "importance": lgb_model.feature_importances_
    }
).sort_values(["importance"])
fig = px.bar(importance, x="col", y="importance")
fig.show()
# %%
# predict
res_df = pd.DataFrame(
    data={
        "num_sold": lgb_model.predict(test_data)
    },
    index=test_data.index
)
sub_df = pd.read_csv(CFG.sample_path, usecols=["row_id"])
sub_df = sub_df.merge(
    res_df,
    left_on="row_id",
    right_index=True
)
print(sub_df.head(10))
sub_df.to_csv(CFG.result_folder / "result_submission.csv", index=False)
# %%
