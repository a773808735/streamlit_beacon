import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objs as go
import numpy as np


@st.cache_data
def load_data():
    # Make sure the file path is correct
    df = pd.read_csv("github_dataset.csv")
    return df.copy()


def intro():
    st.write("# Welcome to Analysis Page! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a langiage from the dropdown on the left** to see some how the
        language performs!

        ### Get Original data

        - Use your own data to [get some interesting finding!
          ](https://www.kaggle.com/datasets/nikhil25803/github-dataset/data)
        - Explore a [more complex dataset](https://www.kaggle.com/datasets/nikhil25803/github-dataset/data)
    """
    )


def repo_activity_analysis(df, top_n=10):
    st.markdown("## Repository Activity Analysis🏃‍")
    df["activity_score"] = df["issues_count"] + df["pull_requests"] + df["contributors"]

    top_active_repos = df.nlargest(top_n, "activity_score")

    fig = px.bar(
        top_active_repos,
        x="repositories",
        y="activity_score",
        color="language",
        title=f"Top {top_n} Active Repositories",
    )
    st.plotly_chart(fig, use_container_width=True)

    language_distribution = top_active_repos["language"].value_counts().reset_index()
    language_distribution.columns = ["language", "count"]
    fig_language = px.pie(
        language_distribution,
        names="language",
        values="count",
        title="Language Distribution in Top Active Repositories",
    )
    st.plotly_chart(fig_language, use_container_width=True)


def repo_treasure_analysis(df, top_n=10):
    st.markdown(
        """
                ## Repository Treasure Analysis🤩🤩
                ### Here the analysis helps us to find treasure repository
                - treasure score: star * pull 
    """
    )

    df["treasure_score"] = 1.0 * df["pull_requests"] * df["stars_count"]

    top_active_repos = df.nlargest(top_n, "treasure_score")

    fig = px.bar(
        top_active_repos,
        x="repositories",
        y="treasure_score",
        color="language",
        title=f"Top {top_n} Treasure Repositories",
    )
    st.plotly_chart(fig, use_container_width=True)

    language_distribution = top_active_repos["language"].value_counts().reset_index()
    language_distribution.columns = ["language", "count"]
    fig_language = px.pie(
        language_distribution,
        names="language",
        values="count",
        title="Language Distribution in Top Treasure Repositories",
    )
    st.plotly_chart(fig_language, use_container_width=True)


def single_language_activity_analysis(df_language, top_n=10):
    st.markdown("## Repository Activity Analysis🏃‍")
    df_language["activity_score"] = (
        df_language["issues_count"]
        + df_language["pull_requests"]
        + df_language["contributors"]
    )

    # 选取活跃度最高的前N个仓库
    top_active_repos = df_language.nlargest(top_n, "activity_score")

    # 可视化最活跃的仓库
    fig = px.bar(
        top_active_repos,
        x="repositories",
        y="activity_score",
        color="language",
        title=f"Top {top_n} Active Repositories",
    )
    st.plotly_chart(fig, use_container_width=True)


def single_language_treasure_analysis(df_language, top_n=10):
    st.markdown(
        """
                ## Repository Treasure Analysis🤩🤩
                ### Here the analysis helps us to find treasure repository
                - treasure score: star * pull 
    """
    )
    df_language["activity_score"] = (
        df_language["pull_requests"] * df_language["stars_count"]
    )

    # 选取活跃度最高的前N个仓库
    top_active_repos = df_language.nlargest(top_n, "activity_score")

    # 可视化最活跃的仓库
    fig = px.bar(
        top_active_repos,
        x="repositories",
        y="activity_score",
        color="language",
        title=f"Top {top_n} Treasure Repositories",
    )
    st.plotly_chart(fig, use_container_width=True)


def plotting_demo(name):
    st.markdown(f"# {name}")
    df = load_data()
    df.dropna(inplace=True)
    if name != "All Language":
        df = df[df["language"] == name]

    # 寻找最受欢迎的项目，并使用适当形式展示
    max_star = df["stars_count"].max()
    max_df = df[df["stars_count"] == max_star].iloc[0]
    st.write("## 👑👑Most Popular Project is...")
    st.write(f"#### {max_df['repositories']}")
    # Calculate and display the average stars and forks
    mean_star = df["stars_count"].mean()
    mean_fork = df["forks_count"].mean()
    mean_issue = df["issues_count"].mean()
    mean_pull = df["pull_requests"].mean()
    mean_con = df["contributors"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("stars", max_df["stars_count"], max_df["stars_count"] - mean_star)
    col2.metric("forks", max_df["forks_count"], max_df["forks_count"] - mean_fork)
    col3.metric("issues", max_df["issues_count"], max_df["issues_count"] - mean_issue)
    col4.metric("pulls", max_df["pull_requests"], max_df["pull_requests"] - mean_pull)
    col5.metric(
        "contributors", max_df["contributors"], max_df["contributors"] - mean_con
    )

    st.write(f"#### average repository")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("stars", "{:.2f}".format(mean_star))
    col2.metric("forks", "{:.2f}".format(mean_fork))
    col3.metric("issues", "{:.2f}".format(mean_issue))
    col4.metric("pulls", "{:.2f}".format(mean_pull))
    col5.metric("contributors", "{:.2f}".format(mean_con))

    st.write("## Raw data virolization")
    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()  # Gets all numeric columns

    metric_select = st.selectbox("Choose the metric to display", num_cols)

    fig_metric = px.histogram(
        df, x=metric_select, nbins=20, title=f"Histogram of {metric_select}"
    )
    fig_metric.update_layout(title_x=0.4)
    st.plotly_chart(fig_metric, use_container_width=True)

    st.markdown(
        """
        - From this diagram, we can find most of the repositories are small repositories, 
        with a character of low star/fork/contributor/issues😢. 
        - In common sense, this might be a negative feedback. But in Github, this doesn't
        make sense, every programmer keeps building their repository for fun😆!
    """
    )

    if name == "All Language":
        st.write("## Language Statistics✍️")

        language_stats = (
            df.groupby("language")
            .agg({"stars_count": "mean", "forks_count": "mean", "language": "count"})
            .rename(columns={"language": "repo_count"})
            .reset_index()
        )

        # 创建双轴图
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=language_stats["language"],
                y=language_stats["stars_count"],
                name="Average Stars",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=language_stats["language"],
                y=language_stats["forks_count"],
                name="Average Forks",
                yaxis="y2",
            )
        )

        # 设置Y轴2
        fig.update_layout(
            yaxis2=dict(title="Average Forks", overlaying="y", side="right"),
            yaxis=dict(title="Average Stars"),
            xaxis_title="Programming Language",
            title="Average Stars and Forks by Programming Language",
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            Though some small language don't have many appearances, they are stars maker!⭐
        """
        )

    st.write("## Language Statistics✍️")
    top_repos = df.nlargest(15, "contributors")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_repos["repositories"],
            y=top_repos["contributors"],
            name="Contributors",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=top_repos["repositories"],
            y=top_repos["issues_count"],
            name="Issues",
            yaxis="y2",
        )
    )

    # 设置Y轴2
    fig.update_layout(
        yaxis2=dict(title="Issues", overlaying="y", side="right"),
        yaxis=dict(title="Contributors"),
        xaxis_title="Repositories",
        title=f"Top {15} Repositories by Contributors in {name}, issues",
        title_x=0.2,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        Generally, more contributors more issues!🤣
    """
    )

    fig_1 = go.Figure()
    fig_1.add_trace(
        go.Bar(
            x=top_repos["repositories"],
            y=top_repos["contributors"],
            name="Contributors",
        )
    )
    fig_1.add_trace(
        go.Scatter(
            x=top_repos["repositories"],
            y=top_repos["forks_count"],
            name="Forks",
            yaxis="y2",
        )
    )

    # 设置Y轴2
    fig_1.update_layout(
        yaxis2=dict(title="Forks", overlaying="y", side="right"),
        yaxis=dict(title="Contributors"),
        xaxis_title="Repositories",
        title=f"Top {15} Repositories by Contributors in {name}, forks",
        title_x=0.2,
    )

    st.plotly_chart(fig_1, use_container_width=True)

    st.markdown(
        """
        There could be many forks though few contributors.
    """
    )

    if name == "All Language":
        repo_activity_analysis(df)
        repo_treasure_analysis(df)

    else:
        single_language_activity_analysis(df)
        single_language_treasure_analysis(df)


page_names_to_funcs = {
    "—": intro,
    "All Language": plotting_demo,
    "JavaScript": plotting_demo,
    "Python": plotting_demo,
    "Java": plotting_demo,
    "C++": plotting_demo,
    "C": plotting_demo,
    "PHP": plotting_demo,
    "Rust": plotting_demo,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
if demo_name == "—":
    page_names_to_funcs[demo_name]()
else:
    page_names_to_funcs[demo_name](demo_name)
