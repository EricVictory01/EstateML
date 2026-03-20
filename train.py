import joblib
from sklearn.cluster import KMeans
from preprocessing import load_and_clean_data, preprocess_features

def train():
    df_new = load_and_clean_data("data/nigeria_houses_data.csv")

    df_new, X_scaled, scaler, le_state, le_town, le_title, feature_cols = preprocess_features(df_new)

    # Optimal K = 4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_new['cluster'] = kmeans.fit_predict(X_scaled)

    # Save everything in the models folder
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le_state, "models/le_state.pkl")
    joblib.dump(le_town, "models/le_town.pkl")
    joblib.dump(le_title, "models/le_title.pkl")

    df_new.to_csv("data/nigeria_houses_clustered.csv", index=False)

    print("Training complete and artifacts saved.")


if __name__ == "__main__":
    train()