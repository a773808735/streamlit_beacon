import streamlit as st
import pandas as pd
import plotly.express as px
import time


@st.cache_data
def load_data():
    # Make sure the file path is correct
    df = pd.read_csv("github_dataset.csv")
    return df.copy()


st.title("Github Data Overview")
st.write("# Welcome to Github Data Show! 👋")
st.markdown(
    """
    Github is a website for hackers to share their ideas and collaborate with each other.

    **🏠 Repository:** This is the main page of our code, where we store our codes.
    
    **⭐ Star:** If you like this repository, don't hesitate to give it a star!
    
    **🍴 Fork:** We need this request to develop code on our own PC!
    
    **❌ Issues:** Some questions about code or some bugs. 🪲
    
    **🧵 Pull:** Used to combine two different branches/bases.
    
    **👨‍💻 Contributor:** Our contributor team!
    
    **🦀 Programming Language:**
    
    ### Raw Data

    Let's have an overview of the raw data!
    """
)

data = load_data()
data = data.dropna(subset=["language"])
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f"Completing {i+1}%")
    bar.progress(i + 1)
    time.sleep(0.02)
language_counts = data["language"].value_counts().reset_index()
language_counts.columns = ["language", "count"]
st.write(data)

fig = px.pie(
    language_counts[:15], values="count", names="language", title="Repository Languages"
)
fig.update_layout(
    title_x=0.2,
    legend=dict(orientation="h", yanchor="bottom", y=-1.2, xanchor="center", x=0.5),
)

top_repos = data.sort_values(by="stars_count", ascending=False).head(5)
custom_ticktext = [f"repo{i+1}" for i in range(top_repos.shape[0])]
right_fig = px.bar(
    top_repos,
    x="repositories",
    y=["stars_count", "forks_count"],
    labels={"value": "Count", "variable": "Type"},
    title="Top 5 Repositories by Stars and their Forks",
    barmode="group",
)
right_fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=top_repos["repositories"],
        ticktext=custom_ticktext,
    ),
    title_x=0,
)

left_column, right_column = st.columns(2)
with left_column:
    st.plotly_chart(fig, use_container_width=True)

with right_column:
    st.plotly_chart(right_fig, use_container_width=True)

st.markdown(
    """
    ### See more detail analysis by clicking the sidebar! 🖥️
    """
)
