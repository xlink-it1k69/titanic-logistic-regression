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