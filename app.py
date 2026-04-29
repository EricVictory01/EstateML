import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from logic import recommend_by_cluster

st.set_page_config(
    page_title="EstateML",
    page_icon="🏠",
    layout="wide"
)

df_filtered = pd.read_csv("data/filtered_properties.csv")

st.title("Smart Property Finder")
st.markdown("###  AI-Powered •  Smart Matching •  Data-Driven")
st.markdown("""
## Smart Property Finder

EstateML helps users discover better property options using machine learning-powered recommendations, match scoring, price insights, and fairness checks.

Choose your preferences below and get smart recommendations instantly.
""")

st.markdown("---")

st.subheader("Set Your Property Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox("State", sorted(df_filtered["state"].unique()))
    town = st.selectbox(
        "Town",
        sorted(df_filtered[df_filtered["state"] == state]["town"].unique())
    )
    title = st.selectbox("Property Type", sorted(df_filtered["title"].unique()))

with col2:
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    toilets = st.slider("Toilets", 1, 10, 2)

with col3:
    parking = st.slider("Parking Spaces", 0, 10, 1)
    budget_min = st.number_input("Minimum Budget (₦)", value=20_000_000, step=1_000_000)
    budget_max = st.number_input("Maximum Budget (₦)", value=100_000_000, step=1_000_000)

top_k = st.slider("Number of Recommendations", 3, 20, 5)

st.markdown("---")

st.markdown("""
### How EstateML Works
1. You enter your property preferences  
2. The system compares your request with similar properties  
3. EstateML returns ranked recommendations with match and price insights  
""")

find_button = st.button("Get Smart Recommendations", use_container_width=True)

if find_button:
    if budget_min >= budget_max:
        st.error("Maximum budget must be greater than minimum budget.")
    else:
        results = recommend_by_cluster(
            bedrooms, bathrooms, toilets, parking,
            budget_min, budget_max,
            state, town, title, top_k
        )

        if not results.empty:
            st.success("Recommendations generated successfully.")

            if "distance" in results.columns and results["distance"].max() != 0:
                results["Match Score"] = (
                    1 - (results["distance"] / results["distance"].max())
                ) * 100
                results["Match Score"] = results["Match Score"].round(1).astype(str) + "%"

            st.info(
                "These properties were selected because they closely match your preferred location, budget, and property features."
            )

            st.subheader("Recommended Properties")
            st.dataframe(results, use_container_width=True)

            if "price" in results.columns:
                st.subheader("Price Distribution")
                fig, ax = plt.subplots()
                ax.hist(results["price"], bins=10)
                ax.set_xlabel("Price (₦)")
                ax.set_ylabel("Number of Properties")
                st.pyplot(fig)

        else:
            st.warning("No matching properties found. Try increasing your budget or changing your preferences.")
