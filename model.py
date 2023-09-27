import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from catboost.utils import get_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


KNOWN_CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "poutcome",
    "age_cat",
    "previous_cat",
    "housing",
    "loan",
    "default",
    "campaign",
    "previous"
]


def _balance_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Balance data by downsampling the majority class

    Args:
        data: Pandas DataFrame with features and target y

    Returns:
        A Pandas DataFrame with balanced data
    """
    logger.info("Balance data.")

    # Separate majority and minority classes
    df_majority = data[data['y'] == 0]
    df_minority = data[data['y'] == 1]

    # Downsample majority class
    df_majority_downsampled = df_majority.sample(n=df_minority.shape[0])

    # Combine minority class with downsampled majority class
    data = pd.concat([df_majority_downsampled, df_minority])

    logger.info(f"Minority class rate: {(round(data[data['y'] == 1].shape[0] / data.shape[0], 2) * 100)}%")

    return data


def _select_features(train_dataset: Pool,
                     test_dataset: Pool,
                     algorithm: EFeaturesSelectionAlgorithm,
                     steps: int = 3) -> list:
    """
    Select features with CatBoost's feature selector

    Args:
        train_dataset: CatBoost Pool with train data
        test_dataset: CatBoost Pool with test data
        algorithm: Algorithm to use for feature selection
        steps: Number of steps to use for feature selection

    Returns:
        A list with selected features
    """
    logger.info("Select features.")

    model = CatBoostClassifier(iterations=500, random_seed=0)
    summary = model.select_features(
        train_dataset,
        eval_set=test_dataset,
        features_for_select=list(range(train_dataset.num_col())),
        num_features_to_select=5,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        logging_level='Silent'
    )
    logger.info('Selected features:', summary['selected_features_names'])

    return summary['selected_features_names']


def _train_model(data: pd.DataFrame) -> CatBoostClassifier:
    """
    Train a CatBoost model with categorical features

    Args:
        data: Pandas DataFrame with features and target y

    Returns:
        A trained classifier model
    """
    logger.info("Train the model.")

    # Select features
    train_labels = data["y"]
    train_data = data.drop("y", axis=1)
    cat_cols = [x for x in train_data.columns if x in KNOWN_CATEGORICAL_FEATURES]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    train_dataset = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    test_dataset = Pool(data=X_test, label=y_test, cat_features=cat_cols)
    selected_features = _select_features(train_dataset, test_dataset, EFeaturesSelectionAlgorithm.RecursiveByShapValues)
    logger.info(f"Train data shape: {X_train.shape[0]}")
    logger.info(f"Test data shape: {X_test.shape[0]}")

    # Train model with selected features
    data_shuffled = data.sample(frac=1, random_state=42)
    train_labels = data_shuffled["y"]
    train_data = data_shuffled[selected_features]
    cat_cols = [x for x in train_data.columns if x in KNOWN_CATEGORICAL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    train_dataset = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    test_dataset = Pool(data=X_test, label=y_test, cat_features=cat_cols)
    logger.info(f"Train data shape: {X_train.shape[0]}")
    logger.info(f"Test data shape: {X_test.shape[0]}")

    model = CatBoostClassifier(iterations=500, learning_rate=0.01, eval_metric='AUC')
    model.fit(train_dataset,
              use_best_model=True,
              eval_set=test_dataset,
              verbose=False)

    # Get model metrics
    logger.info(f"Model params: {model.get_params()}")

    feature_importance = model.feature_importances_
    sorted_idx = np.flip(np.argsort(feature_importance))
    cols_feature_importance = dict(zip(X_test.columns[sorted_idx], feature_importance[sorted_idx]))
    logger.info(f"Feature importance: {cols_feature_importance}")

    cm = get_confusion_matrix(model, test_dataset)
    logger.info(f"Confusion matrix: {cm}")

    logger.info(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
    logger.info(f"Precision: {precision_score(y_test, model.predict(X_test))}")
    logger.info(f"Recall: {recall_score(y_test, model.predict(X_test))}")
    logger.info(f"F1: {f1_score(y_test, model.predict(X_test))}")

    return model


def _save_model(model: CatBoostClassifier, path: Path) -> None:
    """
    Save a trained model

    Args:
        model: A trained CatBoost model
        path: Path to save the model
    """
    logger.info("Save the model.")

    assert isinstance(model, CatBoostClassifier), "Model must be a CatBoostClassifier instance"
    assert isinstance(path, Path), "Path must be a Path instance"

    model.save_model(path)


def model_pipeline(
        data: pd.DataFrame,
        save_model: bool = False,
) -> None:
    data_balanced = _balance_data(data)
    model = _train_model(data_balanced)
    if save_model:
        _save_model(model, Path("data/output/model.cbm"))
