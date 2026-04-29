import streamlit as st
import pandas as pd
from logic import recommend_by_cluster

# Load original filtered dataset (for dropdowns)
df_filtered = pd.read_csv("data/filtered_properties.csv")

st.title("Smart Property Finder")
st.markdown("""
### Find your perfect home, whether in Lagos or Abuja!
This System uses a **K-Means Clustering** Machine Learning model to analyze property patterns and recommend the perfect home for you.
Enter your preferences at the sidebar, and we'll find properties that fit what you are looking for.
""")
with st.sidebar:
    st.header("Your Preferences")

    bedrooms  = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    toilets   = st.slider("Toilets", 1, 10, 2)
    parking   = st.slider("Parking Spaces", 0, 10, 1)

    budget_min = st.number_input("Min Budget", value=20_000_000)
    budget_max = st.number_input("Max Budget", value=100_000_000)

    state = st.selectbox("State", sorted(df_filtered['state'].unique()))
    town  = st.selectbox(
        "Town",
        sorted(df_filtered[df_filtered['state'] == state]['town'].unique())
    )
    title = st.selectbox("Property Type", sorted(df_filtered['title'].unique()))

    top_k = st.slider("Results", 3, 20, 5)

if st.button("Find Properties"):
    results = recommend_by_cluster(
        bedrooms, bathrooms, toilets, parking,
        budget_min, budget_max,
        state, town, title, top_k
    )

    if not results.empty:
        st.dataframe(results)
    else:
        st.warning("No results found.")
