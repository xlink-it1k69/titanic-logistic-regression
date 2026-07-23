import pandas as pd


def load_data(filepath):
    """
    Load Titanic dataset.

    Parameters
    ----------
    filepath : str
        Path to CSV file.

    Returns
    -------
    pandas.DataFrame
    """

    df = pd.read_csv(filepath)

    return df

def check_missing_values(df):
    """
    Check missing values.
    """

    missing = df.isnull().sum()

    return missing

def missing_percentage(df):

    missing = df.isnull().sum()

    percent = missing / len(df) * 100

    return percent.sort_values(ascending=False)

def drop_columns(df):
    """
    Remove columns that will not be used.
    """

    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    return df.drop(columns=columns_to_drop)

def fill_missing_values(df):
    """
    Fill missing values.
    """

    df = df.copy()

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    return df

def encode_sex(df):
    """
    Encode Sex column.
    """

    df = df.copy()

    mapping = {
        "male": 0,
        "female": 1
    }

    df["Sex"] = df["Sex"].map(mapping)

    return df

def encode_embarked(df):
    """
    Encode Embarked column.
    """

    df = df.copy()

    mapping = {
        "S": 0,
        "C": 1,
        "Q": 2
    }

    df["Embarked"] = df["Embarked"].map(mapping)

    return df