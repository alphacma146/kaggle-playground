# %%
# Standerd lib
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna.integration.lightgbm as opt_lgb
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.linear_model import HuberRegressor, LogisticRegression
from tqdm import tqdm
import plotly.express as px

PARAM_SEARCH = True


@dataclass
class Config:
    train_path: Path = Path(r"Data\train.csv")
    test_path: Path = Path(r"Data\test.csv")
    sample_path: Path = Path(r"Data\sample_submission.csv")
    result_folder: Path = Path(r"Data")
    category_params: list = field(default_factory=lambda: [
        'measurement_0', 'measurement_1', 'measurement_2'
    ])
    model_params: dict = field(default_factory=lambda: {
        "device": "gpu",
        "objective": "binary",
        "boosting_type": "gbdt",
        "max_bins": 255,
        "learning_rate": 0.05
    })
    corr_dict: dict = field(
        default_factory=lambda: {
            'measurement_15': {
                'A': ['measurement_11', 'measurement_12'],
                'B': ['measurement_10', 'measurement_11', 'measurement_13'],
                'C': ['measurement_12', 'measurement_14'],
                'D': ['measurement_10', 'measurement_12', 'measurement_14'],
                'E': [],
                'F': ['measurement_11'],
                'G': ['measurement_10', 'measurement_11', 'measurement_14'],
                'H': ['measurement_10', 'measurement_11', 'measurement_12'],
                'I': []
            },
            'measurement_16': {
                'A': [
                    'measurement_11',
                    'measurement_12',
                    'measurement_14',
                    'measurement_15'
                ],
                'B': ['measurement_10', 'measurement_13', 'measurement_15'],
                'C': ['measurement_11', 'measurement_14', 'measurement_15'],
                'D': ['measurement_10', 'measurement_12', 'measurement_15'],
                'E': [
                    'measurement_10',
                    'measurement_11',
                    'measurement_12',
                    'measurement_14'
                ],
                'F': [
                    'measurement_10',
                    'measurement_11',
                    'measurement_13',
                    'measurement_15'
                ],
                'G': ['measurement_10', 'measurement_11', 'measurement_14'],
                'H': ['measurement_10', 'measurement_11', 'measurement_15'],
                'I': [
                    'measurement_11',
                    'measurement_12',
                    'measurement_13',
                    'measurement_14'
                ]
            },
            'measurement_17': {
                'A': [
                    'measurement_4',
                    'measurement_5',
                    'measurement_6',
                    'measurement_7',
                    'measurement_8'
                ],
                'B': [
                    'measurement_3',
                    'measurement_4',
                    'measurement_5',
                    'measurement_7',
                    'measurement_9'
                ],
                'C': [
                    'measurement_5',
                    'measurement_7',
                    'measurement_8',
                    'measurement_9'
                ],
                'D': [
                    'measurement_3',
                    'measurement_5',
                    'measurement_6',
                    'measurement_7',
                    'measurement_8'
                ],
                'E': [
                    'measurement_4',
                    'measurement_5',
                    'measurement_6',
                    'measurement_8',
                    'measurement_9'
                ],
                'F': [
                    'measurement_4',
                    'measurement_5',
                    'measurement_6',
                    'measurement_7'
                ],
                'G': [
                    'measurement_4',
                    'measurement_5',
                    'measurement_6',
                    'measurement_8',
                    'measurement_9'
                ],
                'H': [
                    'measurement_4',
                    'measurement_5',
                    'measurement_7',
                    'measurement_8',
                    'measurement_9'
                ],
                'I': [
                    'measurement_3',
                    'measurement_4',
                    'measurement_7',
                    'measurement_8',
                    'measurement_9'
                ]
            },
        })


CFG = Config()
# %%
# preprocess


def huberregression(data: pd.DataFrame,) -> pd.DataFrame:

    def fit_model(
        x,
        y,
        start_eps=1.0,
        end_eps=10.0,
        step=1
    ):
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2
        )
        res_score = []
        for eps in np.arange(start_eps, end_eps + step, step):
            model = HuberRegressor(epsilon=eps)
            model.fit(x_train, y_train)
            res = mean_squared_error(y_valid, model.predict(x_valid))
            res_score.append((model, eps, res))
        best_model, best_eps, best_score = min(res_score, key=lambda x: x[2])
        if step > 0.001:
            print(step, best_eps, best_score)
            st_eps = best_eps - step if best_eps - step > 1 else 1
            best_model, best_eps, best_score = fit_model(
                x, y, st_eps, best_eps + step, np.true_divide(step, 10)
            )

        return best_model, best_eps, best_score

    for pcode in data["product_code"].unique():
        p_data = data.query(f"product_code=='{pcode}'")
        for fill_col in CFG.corr_dict.keys():
            col_list = CFG.corr_dict[fill_col][pcode]
            if len(col_list) == 0:
                continue
            train = p_data.query(f"{fill_col}=={fill_col}")
            test = p_data.query(f"{fill_col}!={fill_col}")[col_list]
            valid = train[fill_col].copy()
            train = train[col_list].fillna(train[col_list].median()).copy()
            _, eps, _ = fit_model(train, valid)
            model = HuberRegressor(epsilon=eps)
            model.fit(train, valid)
            data.loc[test.index, fill_col] = model.predict(
                test.fillna(test.median())
            )

    return data


def preprocess(data: pd.DataFrame) -> pd.DataFrame:

    data.set_index("id", drop=True, inplace=True)
    use_col = [
        col for col in data.columns
        if col.startswith("measurement")
    ]
    cat_col = CFG.category_params
    num_col = [col for col in use_col if col not in cat_col]
    data[["attribute_0", "attribute_1"]] = (
        data[["attribute_0", "attribute_1"]]
        .applymap(lambda x: int(x.split("_")[-1]))
    )
    endu = data["attribute_0"] * data["attribute_1"]
    area = data["attribute_2"] * data["attribute_3"]
    data = huberregression(data)
    data = data[use_col + ["loading"]].copy()
    for alpha, beta in (
        (0, 1),
        (0, 2),
        (1, 2)
    ):
        data[f"div_meas_{alpha}_{beta}"] = (
            data[f"measurement_{alpha}"]
            / (data[f"measurement_{beta}"] + 1)
        )
        data[f"sub_meas_{alpha}_{beta}"] = (
            data[f"measurement_{alpha}"]
            - data[f"measurement_{beta}"]
        )
    num_data = data[num_col].aggregate(
        ["mean", "std", "min", "max"], axis=1
    )
    cat_data = pd.concat(
        [
            data[cat_col].apply(np.sum, axis=1),
            data[cat_col].apply(lambda x: np.log(np.prod(x + 1)), axis=1),
        ],
        axis=1
    )
    cat_data.columns = ["sum", "prod"]
    for col in num_data.columns:
        data[f"meas_num_{col}"] = num_data[col]
    for col in cat_data.columns:
        data[f"meas_cat_{col}"] = cat_data[col]
    for col in use_col + ["loading"]:
        data[f"miss_{col}"] = data[col].isna().astype(int)
    data["total_missing_value"] = data[
        [col for col in data.columns if col.split("_")[0] == "miss"]
    ].sum(axis=1)
    data["log_loading"] = np.log(data["loading"])
    data["area_capacity"] = np.log(data["loading"] / area)
    data["endurance"] = np.log(data["loading"] / endu)

    return data


ss = StandardScaler()
train_data = pd.read_csv(CFG.train_path)
test_data = pd.read_csv(CFG.test_path)
train_labels = train_data["failure"].copy()
meas_col = [
    col for col in train_data.columns
    if col.startswith("measurement") and col not in CFG.category_params
]
ss.fit(train_data[meas_col])
train_data[meas_col] = ss.transform(train_data[meas_col])
test_data[meas_col] = ss.transform(test_data[meas_col])
train_data = preprocess(train_data)
test_data = preprocess(test_data)
print("train_data:", train_data.shape)
print("test_data: ", test_data.shape)
# %%
# tuning
match PARAM_SEARCH:
    case True:
        train_set = lgb.Dataset(train_data, train_labels)
        tuner = opt_lgb.LightGBMTunerCV(
            CFG.model_params,
            train_set=train_set,
            num_boost_round=500,
            folds=RepeatedKFold(n_splits=3, n_repeats=1, random_state=37),
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
            'lambda_l1': 0.23780855594894879,
            'lambda_l2': 0.0011435326967608513,
            'num_leaves': 2,
            'feature_fraction': 0.52,
            'bagging_fraction': 0.44168344717293895,
            'bagging_freq': 2,
            'min_child_samples': 20
        }
        # roc_auc Score: 0.5884839982766031
# %%
# lightgbm model
params |= CFG.model_params
res_score = []
kf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=37)
for tr_idx, va_idx in tqdm(kf.split(train_data)):
    tr_x, tr_y = train_data.iloc[tr_idx], train_labels.iloc[tr_idx]
    va_x, va_y = train_data.iloc[va_idx], train_labels.iloc[va_idx]
    lgb_model = lgb.LGBMClassifier(**params, num_iterations=500)
    lgb_model.fit(
        tr_x, tr_y,
        eval_set=[(va_x, va_y)],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0),
        ]
    )
    res_score.append(
        roc_auc_score(va_y, lgb_model.predict_proba(va_x)[:, 1])
    )

print(params, f"roc_auc Score: {np.mean(res_score)}", sep="\n")
lgb_model = lgb.LGBMClassifier(**params, num_iterations=1000)
lgb_model.fit(
    train_data,
    train_labels,
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
# logistic regression
nona_train = train_data.fillna(train_data.median())
params = {"max_iter": 500, "C": 0.05, "penalty": "l1", "solver": "liblinear"}
res_score = []
kf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=37)
for tr_idx, va_idx in tqdm(kf.split(train_data)):
    tr_x, tr_y = nona_train.iloc[tr_idx], train_labels.iloc[tr_idx]
    va_x, va_y = nona_train.iloc[va_idx], train_labels.iloc[va_idx]
    lgs_model = LogisticRegression(**params)
    lgs_model.fit(tr_x, tr_y)
    res_score.append(
        roc_auc_score(va_y, lgs_model.predict_proba(va_x)[:, 1])
    )
lgs_model = LogisticRegression(**params)
lgs_model.fit(nona_train, train_labels)
print(f"Logistic:{np.mean(res_score)}")
importance = pd.DataFrame(
    data={
        "col": train_data.columns,
        "importance": np.abs(lgs_model.coef_.ravel())
    }
).sort_values(["importance"])
fig = px.bar(importance, x="col", y="importance")
fig.show()
# Logistic:0.5865637546208611
# %%
# predict
lgb_result = lgb_model.predict_proba(test_data)[:, 1]
lgs_result = lgs_model.predict_proba(
    test_data.fillna(test_data.median()))[:, 1]
result = lgb_result * 0.2 + lgs_result * 0.8
res_df = pd.DataFrame(
    data={"failure": result},
    index=test_data.index
)
fig = px.histogram(res_df)
fig.show()
sub_df = pd.read_csv(CFG.sample_path, usecols=["id"])
sub_df = sub_df.merge(res_df, left_on="id", right_index=True)
print(sub_df.head(10))
sub_df.to_csv(CFG.result_folder / "result_submission.csv", index=False)
# %%
