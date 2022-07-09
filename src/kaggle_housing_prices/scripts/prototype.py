"""Prototype of training process."""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from kaggle_housing_prices.pipeline.price_prediction_model import PriceModel
from kaggle_housing_prices.process import feature_engineering, preprocessing, regression
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

# from skopt import BayesSearchCV
# from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).parents[3].resolve()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:5000")
mlflow.set_experiment("regression-prototype-0.1.0-pre-alpha.revision8")

PARAM_GRID = {
    "regression_method": (0, 6 - 1e-3),
    "fit_intercept": (0, 1),
    "alpha": (1e-4, 10),
    "C": (1e-4, 10),
    "kernel": (0, 2 - 1e-3),
    "max_depth": (1, 10),
    "n_estimators": (50, 500),
    "polynomialfeatures__degree": (1, 3),
    "polynomialfeatures__interaction_only": (0, 1),
    "selectfpr__alpha": (0.01, 0.1),
}


def parse_params(input_grid):
    return {
        "regression_method": [
            LinearRegression(),
            Lasso(),
            Ridge(),
            RandomForestRegressor(n_jobs=-1),
            GradientBoostingRegressor(),
            XGBRegressor(),
        ][int(input_grid.get("regression_method"))],
        "fit_intercept": bool(round(input_grid.get("fit_intercept"))),
        "kernel": ["linear", "rbf"][int(input_grid.get("kernel"))],
        "max_depth": round(input_grid.get("max_depth")),
        "n_estimators": round(input_grid.get("n_estimators")),
        "C": input_grid.get("C"),
        "alpha": input_grid.get("alpha")
        if int(input_grid.get("regression_method")) != 4
        else 0.9,
        "polynomialfeatures__degree": round(
            input_grid.get("polynomialfeatures__degree")
        ),
        "polynomialfeatures__interaction_only": bool(
            round(input_grid.get("polynomialfeatures__interaction_only"))
        ),
        "selectfpr__alpha": input_grid.get("selectfpr__alpha"),
    }


def prototype():
    data = pd.read_csv(ROOT_DIR.joinpath("assets", "data", "train.csv").resolve())

    X, y = data.drop("SalePrice", axis=1), data.SalePrice.to_numpy()

    k_folder = RepeatedKFold(n_splits=4, n_repeats=1)

    def optimization_problem(**kwargs):
        with mlflow.start_run(run_name="Optimization Run"):
            parsed = parse_params(kwargs)
            mlflow.log_params(
                {key: str(value)[:250] for key, value in parsed.items()}
            )  # Adding this because XGBoost parameters exceed character limit
            model = PriceModel(
                preprocessor=preprocessing.Preprocessor(),
                feature_engineer=feature_engineering.PolynomialFeatureEngineer(),
                regressor=regression.SklearnRegressor(**parsed),
            )

            reports = []

            _, ax = plt.subplots(figsize=(10, 10))

            for train_index, val_index in k_folder.split(X, y):
                X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
                y_train, y_val = y[train_index], y[val_index]

                model.fit(X_train, y_train, skip_report=True)
                _, report = model.predict(X_val, y_val, skip_report=True)

                reports.append(report)

            run_report = {
                key: np.mean([sub_result[key] for sub_result in reports])
                for key in reports[0].keys()
            }

            # Get residuals distribution
            mean_residuals = np.mean(
                np.hstack([r["residuals"].reshape(-1, 1) for r in reports]), axis=1
            )

            _ = sns.histplot(mean_residuals, ax=ax)
            plt.savefig(
                ROOT_DIR.joinpath("assets", "artifacts", "residuals.png")
                .resolve()
                .as_posix()
            )

            mlflow.log_metrics(run_report)
            mlflow.sklearn.log_model(model, artifact_path="price_model")
            mlflow.log_artifact(
                ROOT_DIR.joinpath("assets", "artifacts", "residuals.png")
                .resolve()
                .as_posix()
            )
            plt.cla()

        return np.mean(run_report["root_msle"] * -1)

    optimizer = BayesianOptimization(
        f=optimization_problem,
        pbounds=PARAM_GRID,
        random_state=1,
        bounds_transformer=SequentialDomainReductionTransformer(),
    )
    optimizer.maximize(init_points=3, n_iter=25)

    best_params = optimizer.max.get("params")
    parsed_best = parse_params(best_params)

    final_model = PriceModel(
        preprocessor=preprocessing.Preprocessor(),
        feature_engineer=feature_engineering.TargetEncodingFeatureEngineer(),
        regressor=regression.SklearnRegressor(**parsed_best),
    )

    final_model.fit(X, y, skip_report=True)

    # log everything
    with mlflow.start_run(run_name="Best Result"):
        mlflow.log_metrics({"root_msle": optimizer.max.get("target") * -1})
        mlflow.log_params(parsed_best)
        mlflow.sklearn.log_model(
            final_model, artifact_path="price_model", registered_model_name="PriceModel"
        )


if __name__ == "__main__":
    prototype()
