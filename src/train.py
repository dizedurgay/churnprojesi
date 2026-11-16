import json
import os
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def ensure_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / "Telco-Customer-Churn.csv"
    if not file_path.exists():
        urlretrieve(DATASET_URL, file_path)
    return file_path


def save_figure(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=200)
    plt.close(fig)


def make_churn_distribution_plot(df: pd.DataFrame):
    plot_df = df.copy()
    if plot_df["Churn"].dtype == object:
        plot_df["Churn"] = plot_df["Churn"].map({"No": 0, "Yes": 1})
    counts = plot_df["Churn"].value_counts().sort_index()
    labels = ["No", "Yes"]
    colors = ["#4C78A8", "#F58518"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors)
    ax.set_title("Churn Sınıf Dağılımı")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Müşteri Sayısı")
    for bar, value in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, str(value), ha="center", va="bottom")
    return fig


def make_contract_churn_plot(df: pd.DataFrame):
    plot_df = df.copy()
    if plot_df["Churn"].dtype == object:
        plot_df["Churn"] = plot_df["Churn"].map({"No": 0, "Yes": 1})
    churn_by_contract = (
        plot_df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0)
    )
    if 1 not in churn_by_contract.columns:
        churn_by_contract[1] = 0.0
    if 0 not in churn_by_contract.columns:
        churn_by_contract[0] = 0.0
    churn_by_contract = churn_by_contract[[0, 1]]
    fig, ax = plt.subplots(figsize=(7, 5))
    churn_by_contract[1].plot(kind="bar", color="#F58518", ax=ax)
    ax.set_title("Sözleşme Türüne Göre Churn Oranı")
    ax.set_xlabel("Sözleşme")
    ax.set_ylabel("Churn = Yes Oranı")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig


def make_tenure_monthlycharges_plot(df: pd.DataFrame):
    plot_df = df.copy()
    if plot_df["Churn"].dtype == object:
        plot_df["Churn"] = plot_df["Churn"].map({"No": 0, "Yes": 1})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(plot_df[plot_df["Churn"] == 1]["tenure"], bins=20, kde=True, color="#F58518", ax=axes[0], alpha=0.8)
    sns.histplot(plot_df[plot_df["Churn"] == 0]["tenure"], bins=20, kde=True, color="#4C78A8", ax=axes[0], alpha=0.8)
    axes[0].set_title("Tenure Dağılımı (Churn'e Göre)")
    axes[0].set_xlabel("Tenure (ay)")
    axes[0].set_ylabel("Sıklık")

    sns.boxplot(data=plot_df, x="Churn", y="MonthlyCharges", palette=["#4C78A8", "#F58518"], ax=axes[1])
    axes[1].set_title("MonthlyCharges Dağılımı (Churn'e Göre)")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Aylık Ücret")
    return fig


def compute_observations(df: pd.DataFrame):
    plot_df = df.copy()
    if plot_df["Churn"].dtype == object:
        plot_df["Churn"] = plot_df["Churn"].map({"No": 0, "Yes": 1})
    churn_total = plot_df["Churn"].value_counts()
    no_ratio = churn_total.get(0, 0) / (churn_total.get(0, 0) + churn_total.get(1, 0))
    yes_ratio = churn_total.get(1, 0) / (churn_total.get(0, 0) + churn_total.get(1, 0))

    contract_rates = (
        plot_df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0)
    )
    monthly_yes = float(contract_rates.get(1, pd.Series(0, index=contract_rates.index)).get("Month-to-month", 0.0))

    tenure_yes = plot_df.loc[plot_df["Churn"] == 1, "tenure"].mean()
    tenure_no = plot_df.loc[plot_df["Churn"] == 0, "tenure"].mean()
    monthly_yes_mean = plot_df.loc[plot_df["Churn"] == 1, "MonthlyCharges"].mean()
    monthly_no_mean = plot_df.loc[plot_df["Churn"] == 0, "MonthlyCharges"].mean()

    observations = [
        {
            "title": "Churn sınıf dağılımı",
            "text": (
                f"Veri setinde churn oranı % {yes_ratio * 100:.1f} iken, ayrılmayan müşteriler % {no_ratio * 100:.1f} seviyesindedir. "
                "Bu dengesizlik, sınıflandırma modelinin doğrulama sırasında recall gibi duyarlı metriklere dikkatle bakmasını gerektirir."
            ),
        },
        {
            "title": "Sözleşme türüne göre churn oranı",
            "text": (
                f"Aylık sözleşmede churn oranı % {monthly_yes * 100:.1f} olarak hesaplanırken, daha uzun vadeli sözleşmelerde bu oran daha düşüktür. "
                "Bu bulgu, sözleşme süresinin müşteri kaybını belirleyen güçlü bir gösterge olduğunu destekler."
            ),
        },
        {
            "title": "Tenure ve MonthlyCharges dağılımı",
            "text": (
                f"Churn eden müşterilerin ortalama tenure değeri {tenure_yes:.1f} ay, ayrılmayanların değeriyse {tenure_no:.1f} ay olarak ölçülmüştür. "
                f"Ayrıca churn eden müşterilerin aylık ücret ortalaması {monthly_yes_mean:.2f} TL, ayrılmayanlarda ise {monthly_no_mean:.2f} TL seviyesindedir."
            ),
        },
    ]
    return observations


def make_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Sayısal Değişkenler Korelasyon Isı Haritası")
    return fig


def make_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics, y_prob, y_pred


def plot_roc_curve(models: dict, X_test: pd.DataFrame, y_test: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Eğrileri")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig


def plot_confusion_matrix(cm, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    return fig


def plot_feature_importance(model, feature_names, title: str):
    importances = model.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
    names = [item[0] for item in feature_importance]
    values = [item[1] for item in feature_importance]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=values, y=names, palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Önem Skoru")
    ax.set_ylabel("Özellik")
    return fig


def main():
    df = pd.read_csv(ensure_dataset())
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = make_preprocessor(X)

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )

    rf_estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 8, 12],
        "min_samples_leaf": [1, 5],
    }
    rf_grid = GridSearchCV(
        estimator=rf_estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )

    lr_pipeline.fit(X_train, y_train)
    rf_grid.fit(preprocessor.fit_transform(X_train), y_train)

    lr_cv_scores = cross_val_score(
        estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        X=preprocessor.fit_transform(X_train),
        y=y_train,
        cv=5,
        scoring="f1",
    )

    lr_metrics, lr_probs, _ = evaluate_model(lr_pipeline, X_test, y_test)
    rf_best = rf_grid.best_estimator_
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_best),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_metrics, rf_probs, _ = evaluate_model(rf_pipeline, X_test, y_test)

    feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    feature_plot = plot_feature_importance(rf_pipeline.named_steps["model"], feature_names, "Random Forest - İlk 15 Özellik Önem Skoru")
    save_figure(feature_plot, "05_feature_importance_rf.png")

    churn_dist_plot = make_churn_distribution_plot(df.assign(Churn=df["Churn"].map({0: "No", 1: "Yes"})))
    save_figure(churn_dist_plot, "01_churn_distribution.png")

    contract_plot = make_contract_churn_plot(df.assign(Churn=df["Churn"].map({0: "No", 1: "Yes"})))
    save_figure(contract_plot, "02_contract_churn_rate.png")

    tenure_plot = make_tenure_monthlycharges_plot(df.assign(Churn=df["Churn"].map({0: "No", 1: "Yes"})))
    save_figure(tenure_plot, "03_tenure_monthlycharges.png")

    corr_plot = make_correlation_heatmap(df.assign(Churn=df["Churn"].map({0: "No", 1: "Yes"})))
    save_figure(corr_plot, "04_correlation_heatmap.png")

    lr_cm = plot_confusion_matrix(np.array(lr_metrics["confusion_matrix"]), "Logistic Regression Confusion Matrix")
    save_figure(lr_cm, "06_lr_confusion_matrix.png")

    rf_cm = plot_confusion_matrix(np.array(rf_metrics["confusion_matrix"]), "Random Forest Confusion Matrix")
    save_figure(rf_cm, "07_rf_confusion_matrix.png")

    roc_plot = plot_roc_curve({"Logistic Regression": lr_pipeline, "Random Forest": rf_pipeline}, X_test, y_test)
    save_figure(roc_plot, "08_roc_curve.png")

    observations = compute_observations(df.assign(Churn=df["Churn"].map({0: "No", 1: "Yes"})))
    comparison = [
        {
            "model": "Logistic Regression",
            "accuracy": float(lr_metrics["accuracy"]),
            "precision": float(lr_metrics["precision"]),
            "recall": float(lr_metrics["recall"]),
            "f1": float(lr_metrics["f1"]),
            "roc_auc": float(lr_metrics["roc_auc"]),
            "cv_f1_mean": float(lr_cv_scores.mean()),
        },
        {
            "model": "Random Forest",
            "accuracy": float(rf_metrics["accuracy"]),
            "precision": float(rf_metrics["precision"]),
            "recall": float(rf_metrics["recall"]),
            "f1": float(rf_metrics["f1"]),
            "roc_auc": float(rf_metrics["roc_auc"]),
            "best_cv_f1": float(rf_grid.best_score_),
            "best_params": rf_grid.best_params_,
        },
    ]

    metrics_payload = {
        "source": DATASET_URL,
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target": "Churn",
            "class_balance": {
                "No": int((df["Churn"] == 0).sum()),
                "Yes": int((df["Churn"] == 1).sum()),
            },
        },
        "observations": observations,
        "cross_validation": {
            "logistic_regression_f1_mean": float(lr_cv_scores.mean()),
            "random_forest_best_cv_f1": float(rf_grid.best_score_),
        },
        "models": {
            "logistic_regression": {
                "params": {
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "random_state": RANDOM_STATE,
                },
                "test_metrics": lr_metrics,
                "cv_f1_mean": float(lr_cv_scores.mean()),
            },
            "random_forest": {
                "params": {
                    "best_params": rf_grid.best_params_,
                    "cv": 5,
                    "scoring": "f1",
                },
                "test_metrics": rf_metrics,
                "best_cv_f1": float(rf_grid.best_score_),
            },
        },
        "comparison_table": comparison,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print("Veri seti indirildi ve işleniyor...")
    print(json.dumps({
        "logistic_regression": {
            "f1": round(lr_metrics["f1"], 4),
            "recall": round(lr_metrics["recall"], 4),
            "roc_auc": round(lr_metrics["roc_auc"], 4),
        },
        "random_forest": {
            "f1": round(rf_metrics["f1"], 4),
            "recall": round(rf_metrics["recall"], 4),
            "roc_auc": round(rf_metrics["roc_auc"], 4),
        },
        "best_rf_params": rf_grid.best_params_,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
