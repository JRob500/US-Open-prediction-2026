"""
train.py
--------
Phase 4: trains the from-scratch Perceptron on Phase 2's feature
output, with a time-based train/test split and full MLflow experiment
tracking.
"""

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.perceptron import Perceptron  # noqa: E402


NUMERIC_DIFF_FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_diff", "rank_points_diff",
    "win_pct_last10_diff", "surface_win_pct_last10_diff",
    "age_diff", "ht_diff", "days_since_last_match_diff",
]
NUMERIC_RAW_FEATURES = [
    "h2h_matches", "h2h_player_a_win_pct",
    "player_a_matches_played", "player_b_matches_played",
    "best_of",
]
BOOL_FEATURES = [
    "player_a_unranked", "player_b_unranked",
    "player_a_returning_from_layoff", "player_b_returning_from_layoff",
]
CATEGORICAL_FEATURES = ["surface", "tourney_level", "round"]

TARGET = "player_a_wins"


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["player_a_elo"] - df["player_b_elo"]
    df["surface_elo_diff"] = df["player_a_surface_elo"] - df["player_b_surface_elo"]
    df["rank_diff"] = df["player_b_rank"] - df["player_a_rank"]
    df["rank_points_diff"] = df["player_a_rank_points"] - df["player_b_rank_points"]
    df["win_pct_last10_diff"] = df["player_a_win_pct_last10"] - df["player_b_win_pct_last10"]
    df["surface_win_pct_last10_diff"] = (
        df["player_a_surface_win_pct_last10"] - df["player_b_surface_win_pct_last10"]
    )
    df["age_diff"] = df["player_a_age"] - df["player_b_age"]
    df["ht_diff"] = df["player_a_ht"] - df["player_b_ht"]
    df["days_since_last_match_diff"] = (
        df["player_a_days_since_last_match"] - df["player_b_days_since_last_match"]
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_cols = NUMERIC_DIFF_FEATURES + NUMERIC_RAW_FEATURES + BOOL_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def time_based_split(df: pd.DataFrame, test_frac: float):
    df = df.sort_values("tourney_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df)} matches ({train_df['tourney_date'].min()} "
          f"-> {train_df['tourney_date'].max()})")
    print(f"Test:  {len(test_df)} matches ({test_df['tourney_date'].min()} "
          f"-> {test_df['tourney_date'].max()})")
    return train_df, test_df


def prepare_xy(df: pd.DataFrame):
    feature_cols = (
        NUMERIC_DIFF_FEATURES + NUMERIC_RAW_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES
    )
    X = df[feature_cols].copy()
    for col in NUMERIC_DIFF_FEATURES + NUMERIC_RAW_FEATURES:
        X[col] = X[col].fillna(0)
    for col in BOOL_FEATURES:
        X[col] = X[col].fillna(False).astype(int)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)
    y = df[TARGET].values
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Phase 4: train + track the perceptron")
    parser.add_argument("--features", required=True)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    print(f"Loading {args.features} ({args.tour.upper()})...")
    df = pd.read_csv(args.features, parse_dates=["tourney_date"])
    print(f"Loaded {len(df)} matches\n")

    df = add_diff_features(df)
    train_df, test_df = time_based_split(df, args.test_frac)

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", Perceptron(learning_rate=args.learning_rate, n_epochs=args.n_epochs)),
    ])

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_param("tour", args.tour)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("n_epochs", args.n_epochs)
        mlflow.log_param("test_frac", args.test_frac)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features_raw", X_train.shape[1])

        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)
        test_proba = pipeline.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_logloss = log_loss(y_test, test_proba)
        test_brier = brier_score_loss(y_test, test_proba)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_log_loss", test_logloss)
        mlflow.log_metric("test_brier_score", test_brier)

        mlflow.sklearn.log_model(pipeline, "model", serialization_format="pickle")

        print(f"\nTrain accuracy: {train_acc:.3f}")
        print(f"Test accuracy:  {test_acc:.3f}")
        print(f"Test log loss:  {test_logloss:.3f}")
        print(f"Test Brier score: {test_brier:.3f}")

        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{args.tour}_pipeline.pkl"
        joblib.dump(pipeline, model_path)
        print(f"\nAlso saved plain pipeline to {model_path}")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
