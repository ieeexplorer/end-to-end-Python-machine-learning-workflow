import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class AppConfig:
    random_state: int = 42
    n_samples: int = 3000
    test_size: float = 0.2
    model_dir: str = "models"
    output_dir: str = "outputs"
    data_dir: str = "data"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
class TicketDataGenerator:
    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.default_rng(random_state)

    def generate(self, n_samples: int) -> pd.DataFrame:
        channels = ["email", "chat", "phone", "web"]
        products = ["billing", "platform", "mobile_app", "hardware", "integration"]
        customer_tiers = ["free", "pro", "enterprise"]
        regions = ["UK", "EU", "US", "APAC"]
        issue_types = ["bug", "question", "incident", "request", "access"]

        df = pd.DataFrame({
            "ticket_id": np.arange(10000, 10000 + n_samples),
            "channel": self.rng.choice(channels, n_samples, p=[0.35, 0.30, 0.15, 0.20]),
            "product": self.rng.choice(products, n_samples),
            "customer_tier": self.rng.choice(customer_tiers, n_samples, p=[0.50, 0.35, 0.15]),
            "region": self.rng.choice(regions, n_samples),
            "issue_type": self.rng.choice(issue_types, n_samples),
            "customer_age_days": self.rng.integers(1, 2000, n_samples),
            "account_value_usd": np.round(self.rng.normal(1200, 900, n_samples).clip(0, None), 2),
            "message_length": self.rng.integers(20, 2000, n_samples),
            "attachments_count": self.rng.integers(0, 6, n_samples),
            "previous_tickets": self.rng.integers(0, 15, n_samples),
            "hours_open": np.round(self.rng.gamma(shape=2.0, scale=10.0, size=n_samples), 2),
            "sentiment_score": np.round(self.rng.uniform(-1, 1, n_samples), 3),
            "error_code_present": self.rng.choice([0, 1], n_samples, p=[0.7, 0.3]),
            "vip_flag": self.rng.choice([0, 1], n_samples, p=[0.9, 0.1]),
        })

        # Feature engineering inputs
        df["is_enterprise"] = (df["customer_tier"] == "enterprise").astype(int)
        df["is_incident"] = (df["issue_type"] == "incident").astype(int)
        df["is_platform_or_integration"] = df["product"].isin(["platform", "integration"]).astype(int)
        df["negative_sentiment_flag"] = (df["sentiment_score"] < -0.35).astype(int)

        # Generate realistic priority target
        priority_score = (
            1.2 * df["is_enterprise"]
            + 1.4 * df["is_incident"]
            + 0.7 * df["vip_flag"]
            + 0.5 * df["error_code_present"]
            + 0.015 * df["hours_open"]
            + 0.06 * df["previous_tickets"]
            + 0.0004 * df["message_length"]
            + 0.9 * df["negative_sentiment_flag"]
            + 0.6 * df["is_platform_or_integration"]
            + self.rng.normal(0, 0.6, n_samples)
        )

        df["priority"] = pd.cut(
            priority_score,
            bins=[-np.inf, 1.8, 3.8, np.inf],
            labels=["low", "medium", "high"],
        ).astype(str)

        # Generate escalation target
        escalation_score = (
            0.9 * (df["priority"] == "high").astype(int)
            + 0.018 * df["hours_open"]
            + 0.08 * df["previous_tickets"]
            + 0.7 * df["negative_sentiment_flag"]
            + 0.5 * df["error_code_present"]
            + 0.8 * df["vip_flag"]
            + self.rng.normal(0, 0.5, n_samples)
        )

        df["escalation_risk"] = (escalation_score > 1.9).astype(int)

        return df


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
class FeatureBuilder:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()

        transformed["msg_len_per_attachment"] = transformed["message_length"] / (transformed["attachments_count"] + 1)
        transformed["account_value_per_ticket"] = transformed["account_value_usd"] / (transformed["previous_tickets"] + 1)
        transformed["customer_maturity"] = pd.cut(
            transformed["customer_age_days"],
            bins=[0, 30, 180, 730, np.inf],
            labels=["new", "growing", "mature", "loyal"],
        ).astype(str)
        transformed["open_time_bucket"] = pd.cut(
            transformed["hours_open"],
            bins=[-np.inf, 4, 24, 72, np.inf],
            labels=["fresh", "same_day", "aged", "critical_delay"],
        ).astype(str)

        return transformed


# -----------------------------------------------------------------------------
# Model service
# -----------------------------------------------------------------------------
class TicketPriorityModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: Pipeline | None = None
        self.feature_columns: list[str] | None = None

    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])

        classifier = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        return pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_columns = X.columns.tolist()
        self.model = self.build_pipeline(X)
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)

    def save(self, model_path: Path) -> None:
        if self.model is None or self.feature_columns is None:
            raise ValueError("Cannot save an untrained model.")

        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, model_path)

    def load(self, model_path: Path) -> None:
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dirs(config: AppConfig) -> None:
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "channel",
        "product",
        "customer_tier",
        "region",
        "issue_type",
        "customer_age_days",
        "account_value_usd",
        "message_length",
        "attachments_count",
        "previous_tickets",
        "hours_open",
        "sentiment_score",
        "error_code_present",
        "vip_flag",
        "is_enterprise",
        "is_incident",
        "is_platform_or_integration",
        "negative_sentiment_flag",
        "msg_len_per_attachment",
        "account_value_per_ticket",
        "customer_maturity",
        "open_time_bucket",
    ]
    X = df[feature_columns].copy()
    y = df["priority"].copy()
    return X, y


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def export_predictions(
    original_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray,
    output_dir: Path,
) -> None:
    pred_df = X_test.copy()
    pred_df["actual_priority"] = y_test.values
    pred_df["predicted_priority"] = y_pred

    classes = ["high", "low", "medium"]
    if proba.shape[1] == 3:
        model_class_order = sorted(np.unique(y_test))
        for idx, class_name in enumerate(model_class_order):
            pred_df[f"proba_{class_name}"] = proba[:, idx]

    pred_df["needs_manager_review"] = (
        (pred_df["predicted_priority"] == "high")
        & (pred_df["hours_open"] > 24)
    ).astype(int)

    pred_df.to_csv(output_dir / "ticket_predictions.csv", index=False)

    top_cases = pred_df.sort_values(
        by=["needs_manager_review", "hours_open"],
        ascending=[False, False],
    ).head(25)
    top_cases.to_json(output_dir / "top_priority_cases.json", orient="records", indent=2)

    n8n_payload = top_cases[[
        "customer_tier",
        "product",
        "issue_type",
        "hours_open",
        "predicted_priority",
        "needs_manager_review",
    ]].to_dict(orient="records")

    with open(output_dir / "n8n_payload.json", "w", encoding="utf-8") as f:
        json.dump(n8n_payload, f, indent=2)


def print_business_summary(df: pd.DataFrame) -> None:
    print("\n=== Business Summary ===")
    print("Priority distribution:")
    print(df["priority"].value_counts(normalize=True).round(3))

    print("\nEscalation risk distribution:")
    print(df["escalation_risk"].value_counts(normalize=True).round(3))

    print("\nAverage hours open by priority:")
    print(df.groupby("priority")["hours_open"].mean().round(2))


# -----------------------------------------------------------------------------
# Main application flow
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Support Ticket Prioritizer")
    parser.add_argument("--samples", type=int, default=3000, help="Number of synthetic samples to generate")
    parser.add_argument("--save-data", action="store_true", help="Save generated dataset to CSV")
    args = parser.parse_args()

    setup_logging()
    config = AppConfig(n_samples=args.samples)
    ensure_dirs(config)

    logging.info("Starting project with config: %s", asdict(config))

    generator = TicketDataGenerator(random_state=config.random_state)
    raw_df = generator.generate(config.n_samples)
    df = FeatureBuilder.transform(raw_df)

    if args.save_data:
        dataset_path = Path(config.data_dir) / "synthetic_tickets.csv"
        df.to_csv(dataset_path, index=False)
        logging.info("Saved synthetic dataset to %s", dataset_path)

    print_business_summary(df)

    X, y = split_features_targets(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model_service = TicketPriorityModel(random_state=config.random_state)
    model_service.fit(X_train, y_train)

    y_pred = model_service.predict(X_test)
    proba = model_service.predict_proba(X_test)

    metrics = evaluate_model(y_test, y_pred)

    print("\n=== Model Accuracy ===")
    print(round(metrics["accuracy"], 4))

    print("\n=== Classification Report ===")
    print(pd.DataFrame(metrics["classification_report"]).transpose().round(3))

    print("\n=== Confusion Matrix ===")
    print(pd.DataFrame(metrics["confusion_matrix"]))

    model_path = Path(config.model_dir) / "ticket_priority_model.joblib"
    model_service.save(model_path)
    logging.info("Saved trained model to %s", model_path)

    export_predictions(
        original_df=df,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        proba=proba,
        output_dir=Path(config.output_dir),
    )
    logging.info("Saved outputs to %s", config.output_dir)

    print("\n=== Files Created ===")
    print(f"- {model_path}")
    print(f"- {Path(config.output_dir) / 'ticket_predictions.csv'}")
    print(f"- {Path(config.output_dir) / 'top_priority_cases.json'}")
    print(f"- {Path(config.output_dir) / 'n8n_payload.json'}")

    print("\nDone. This project simulates an ML-powered ticket triage workflow.")


if __name__ == "__main__":
    main()
