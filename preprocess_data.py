import pandas as pd
from loguru import logger
from ucimlrepo import fetch_ucirepo


def _load_data(load_data_from_local: bool) -> pd.DataFrame:
    """
    Load data from UCI ML Repo (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
    If load_data_from_local is True, load data from local data directory

    Args:
        load_data_from_local: Load data from local or from UCI ML Repo

    Returns:
        Pandas DataFrame with features and target
    """
    logger.info("Load data.")

    if load_data_from_local:
        data = pd.read_csv("data/input/bank-full.csv", sep=";")
        logger.info("Loaded from local.")
    else:
        bank_marketing = fetch_ucirepo(id=222)
        data = bank_marketing['data']['original'].copy()
        logger.info("Loaded from UCI ML Repo.")

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data columns: {data.columns}")

    return data


def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data

    Args:
        data: Pandas DataFrame with features and target

    Returns:
        Cleaned dataframe
    """
    logger.info("Clean data.")

    for col in ["default", "housing", "loan", "y"]:
        data[col].replace({"yes": 1, "no": 0}, inplace=True)

    logger.info(f"Minority class rate: {(round(data[data['y'] == 1].shape[0] / data.shape[0], 2) * 100)}%")

    # Remove outliers from balance and duration
    for col in ["balance", "duration"]:
        q_low = 0
        q_hi = data[col].quantile(0.95)
        data = data[(data[col] < q_hi) & (data[col] > q_low)]

    # Remove previous calls if more than ~4 calls
    q_hi = data["previous"].quantile(0.99)
    data = data[data["previous"] < q_hi]

    # Remove outliers from age
    q_low = 18
    q_hi = data["age"].quantile(0.95)
    data = data[(data["age"] < q_hi) & (data["age"] >= q_low)]

    return data


def _handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values

    Args:
        data: Pandas DataFrame with features and target

    Returns:
        Pandas DataFrame without missing values
    """
    logger.info("Handle missing values.")

    # Fill null values with 'no_previous_contact'
    data["poutcome"].fillna("no_previous_contact", inplace=True)

    # Drop null values
    data.dropna(inplace=True)
    logger.info(f"Data shape: {data.shape}")

    return data


def _analysis(data: pd.DataFrame) -> None:
    """
    Analyze data for initial insights and profiling

    Args:
        data: Pandas DataFrame with features and target
    """
    logger.info("Analyze data.")

    # Some analysis facts
    avg_balance_subs = data[data["y"] == 1]["balance"].mean()
    avg_balance_no_subs = data[data["y"] == 0]["balance"].mean()
    logger.info(f"Average balance for subscribed: {round(avg_balance_subs, 2)}")
    logger.info(f"Average balance for not subscribed: {round(avg_balance_no_subs, 2)}")
    avg_balance_diff = avg_balance_subs - avg_balance_no_subs
    logger.info(f"Average balance difference between subscribed and not subscribed: {avg_balance_diff}")

    avg_age_subs = data[data["y"] == 1]["age"].mean()
    avg_age_no_subs = data[data["y"] == 0]["age"].mean()
    logger.info(f"Average age for subscribed: {round(avg_age_subs, 2)}")
    logger.info(f"Average age for not subscribed: {round(avg_age_no_subs, 2)}")
    avg_age_diff = avg_age_subs - avg_age_no_subs
    logger.info(f"Average age difference between subscribed and not subscribed: {avg_age_diff}")

    avg_duration_subs = data[data["y"] == 1]["duration"].mean()
    avg_duration_no_subs = data[data["y"] == 0]["duration"].mean()
    logger.info(f"Average call duration for subscribed: {round(avg_duration_subs, 2)}")
    logger.info(f"Average call duration for not subscribed: {round(avg_duration_no_subs, 2)}")
    avg_duration_diff = avg_duration_subs - avg_duration_no_subs
    logger.info(
        f"Average call duration difference between subscribed and not subscribed: {avg_duration_diff}")

    housing_load_subs = data[data["housing"] == 1]["y"].mean()
    no_housing_load_subs = data[data["housing"] == 0]["y"].mean()
    logger.info(f"Success rate for clients with housing loan: {round(housing_load_subs, 2)*100}%")
    logger.info(f"Success rate for clients without housing loan: {round(no_housing_load_subs, 2)*100}%")

    first_call_success_rate = data[data["previous_cat"] == 0]["y"].mean()
    logger.info(f"First call success rate: {round(first_call_success_rate, 2)*100}%")

    month_grouped = data.groupby(["month"])["y"].mean().reset_index(name="mean")
    high_success_months = month_grouped.sort_values(by="mean", ascending=False)["month"][:4].tolist()
    logger.info(f"Months with highest success rate: {high_success_months}")

    days_grouped = data.groupby(["day_of_week"])["y"].mean().reset_index(name="mean")
    high_success_days = days_grouped.sort_values(by="mean", ascending=False)["day_of_week"][:10].tolist()
    logger.info(f"Calendar days with highest success rate: {high_success_days}")

    job_grouped = data.groupby(["job"])["y"].mean().reset_index(name="mean")
    high_success_jobs = job_grouped.sort_values(by="mean", ascending=False)["job"][:4].tolist()
    logger.info(f"Jobs with highest success rate: {high_success_jobs}")


def _drop_unavailable_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop features that are not available at the time of prediction

    Args:
        data: Pandas DataFrame with features and target

    Returns:
        Pandas DataFrame without unavailable features
    """
    logger.info("Drop unavailable features.")
    # Drop `duration` that can't be known before a call
    # This attribute highly affects the output target (e.g., if duration=0 then y='no').
    # Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known.
    # Thus, this input should only be included for benchmark purposes and
    # should be discarded if the intention is to have a realistic predictive model.
    if "duration" in data.columns:
        data.drop(["duration"], axis=1, inplace=True)

    return data


def _add_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add features derived from existing features

    Args:
        data: Pandas DataFrame with features and target

    Returns:
        Preprocessed dataframe with added features
    """
    logger.info("Add features to data.")

    # Convert age into categorical variable
    age_categories = ["young_adult", "mid_age_adult_1", "mid_age_adult_2", "mid_age_adult_3", "old_adult"]
    data["age_cat"] = pd.qcut(data["age"], 5, labels=age_categories)

    # How many times was the client contacted before?
    data["previous_cat"] = pd.cut(data["previous"], [-1, 0, 1, 2, 3, 1000], include_lowest=True,
                                  labels=[0, 1, 2, 3, 4])

    # Categorical months to numbers (1-12) for the model to understand the order in months
    data["month_num"] = pd.to_datetime(data["month"], format='%b').dt.month

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data columns: {data.columns}")
    logger.info(f"Minority class rate: {(round(data[data['y'] == 1].shape[0] / data.shape[0], 2) * 100)}%")

    return data


def preprocess_data_pipeline(load_data_from_local: bool = False) -> pd.DataFrame:
    data = _load_data(load_data_from_local)
    data = _clean_data(data)
    data = _handle_missing_values(data)
    data = _add_features(data)
    _analysis(data)
    data = _drop_unavailable_features(data)
    return data
