import pandas as pd
import numpy as np

def load_data(train_path: str, test_path: str):
    '''
    Loads the titanic dataset from csv files and returns train and test dataframes.
    '''

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

def handle_missing_values(df:pd.DataFrame) -> pd.DataFrame:
    """
    - Fill missing Embarked with mode
    - Fill missing Fare with median
    - Drops any extremely missing columns if needed
    """
    if df['Embarked'].isnull().any():
        df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

    # fare
    if df['Fare'].isnull().any():
        df['Fare'].fillna(df['Fare'].median(),inplace=True)


    return df