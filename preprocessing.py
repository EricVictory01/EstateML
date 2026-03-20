import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Create a copy of the original dataset and filter for Abuja and Lagos
    df_new = df.copy()
    df_new['state'] = df_new['state'].str.strip().str.title()
    df_new = df_new[df_new['price'] > 0].dropna()
    df_new = df_new[df_new['state'].isin(['Lagos', 'Abuja'])].copy()

    return df_new

def preprocess_features(df_new):
    # Log transform price
    df_new['log_price'] = np.log1p(df_new['price'])

    # Encoders
    le_state = LabelEncoder()
    le_town  = LabelEncoder()
    le_title = LabelEncoder()

    df_new['state_enc'] = le_state.fit_transform(df_new['state'])
    df_new['town_enc']  = le_town.fit_transform(df_new['town'])
    df_new['title_enc'] = le_title.fit_transform(df_new['title'])

    feature_cols = [
        'bedrooms', 'bathrooms', 'toilets', 'parking_space',
        'log_price', 'state_enc', 'town_enc', 'title_enc'
    ]

    X = df_new[feature_cols].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return df_new, X_scaled, scaler, le_state, le_town, le_title, feature_cols