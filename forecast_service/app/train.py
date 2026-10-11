"""Train the model and persist the resulting artifact."""

from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .model import (
    CAT,
    NUM,
    TARGET,
    load_training_data,
    train_model,
)


ARTIFACTS = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
)


def main() -> None:
    """Train the model, report metrics, and persist the artifact."""

    df = load_training_data()

    # Розбиття даних на train (80%) та test (20%)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    # Навчання моделы виключно на навчальній вибірці
    model = train_model()
    model.fit(
        train_df[NUM + CAT],
        train_df[TARGET],
    )

    # Обчислення метрик на навчальних даних (In-sample)
    train_predictions = model.predict(train_df[NUM + CAT])
    r2_train = r2_score(train_df[TARGET], train_predictions)
    mae_train = mean_absolute_error(train_df[TARGET], train_predictions)

    # Обчислення метрик на тестових даних (Out-of-sample)
    test_predictions = model.predict(test_df[NUM + CAT])
    r2_test = r2_score(test_df[TARGET], test_predictions)
    mae_test = mean_absolute_error(test_df[TARGET], test_predictions)

    print("ML EVALUATION METRICS")
    print("---------------------")
    print(f"[Train / In-Sample]      R2: {r2_train:.4f} | MAE: {mae_train:,.0f} USD")
    print(f"[Test  / Out-of-Sample]  R2: {r2_test:.4f} | MAE: {mae_test:,.0f} USD")
    print("---------------------")

    # Різниця (дельта) для виявлення перенавчання
    r2_diff = r2_train - r2_test
    mae_diff = mae_test - mae_train
    print(f"[Analysis] Delta R2: {r2_diff:+.4f} | Delta MAE: {mae_diff:+,.0f} USD")

    # Збереження навченої моделі
    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path = ARTIFACTS / "model.pkl"

    joblib.dump(
        model,
        artifact_path,
    )

    print(
        f"[train] artifact persisted: {artifact_path}"
    )


if __name__ == "__main__":
    main()