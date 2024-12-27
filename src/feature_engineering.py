import pandas as pd

def fill_missing_ages(df: pd.DataFrame) -> pd.DataFrame:
    """
    function to fill missing Age values using the median age of the passenger's Title group.
    """
    # Extract title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Compute median ages within each Title group
    title_age_map = df.groupby('Title')['Age'].median()
    
    # Fill missing Age
    for title, median_age in title_age_map.items():
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = median_age
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates additional features such as FamilySize, 
    binary Cabin indicator, etc.
    """
    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
    
    # Cabin indicator
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    
    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    
    # One-hot encoding for Embarked
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    return df

def select_model_features(df: pd.DataFrame):
    """
    Selects columns that will be used for model training.
    """
    features = df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin']]
    return features

def preprocess_data(df: pd.DataFrame, is_train: bool = True):
    """
    Full pipeline for data transformations: fill missing values, feature creation, etc.
    If is_train, also drop the 'Survived' from the final dataset if it exists.
    """
    df = fill_missing_ages(df)
    df = create_features(df)
    X = select_model_features(df)
    y = None
    
    if is_train and 'Survived' in df.columns:
        y = df['Survived'].values
    
    return X, y
