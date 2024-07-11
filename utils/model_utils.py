import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold

def process_assignees(df):
    mlb = MultiLabelBinarizer()
    assignees_encoded = mlb.fit_transform(df['assignees'])
    assignees_df = pd.DataFrame(assignees_encoded, columns=mlb.classes_).reset_index(drop=True)
    df = df.drop('assignees', axis=1).reset_index(drop=True)
    assert len(df) == len(assignees_df), "Length of DataFrame and assignees_df do not match!"
    df = pd.concat([df, assignees_df], axis=1).dropna(axis=1, how='all')
    df = df.loc[:, (df != 0).any(axis=0)]
    
    return df

# NOTE: I can add random selection
def balance_dataset(df, random_state=None):
    min_count = df['class'].value_counts().min()
    df_balanced = df.groupby('class').sample(n=min_count, random_state=random_state)
    return df_balanced

# Normlize all float64 and int64 values besides class
def normalize_data(df):
    minmax_scaler = MinMaxScaler()
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.difference(['class'])
    normalized_data = minmax_scaler.fit_transform(df[numeric_features])
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_features)
    normalized_df['class'] = df['class'].values 
    return df

# Remove outlier tasks, drop columns and keep only comments.body 
def filter_data(df):
    df = df[df['assignees'].apply(len) > 0]
    df = df[df['burnedPoints'] <= 10]
    # df = df[df['burnedPoints'] != 0]
    # df = df[df['expectedPoints'] != 1]
    
    df = df.drop(columns=[col for col in ["_id", "createdAt"] if col in df.columns])
    df['comments'] = df['comments'].apply(lambda comments: [comment['body'] for comment in comments] if comments else [])
    return df

# Remove features with low variance and features that are similar 
def remove_redundant_features(df):
    # Identify highly correlated features
    correlation_matrix = df.corr()
    correlated_features = {
        correlation_matrix.columns[i]
        for i in range(len(correlation_matrix.columns))
        for j in range(i)
        if abs(correlation_matrix.iloc[i, j]) > 0.8 # Threshold
    }

    # Identify low variance features (not that important)
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df)
    low_variance_features = set(df.columns[~selector.get_support()])

    df.drop(columns=correlated_features | low_variance_features, inplace=True)
    return df
