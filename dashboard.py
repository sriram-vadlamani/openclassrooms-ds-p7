import requests
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(layout="wide")
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dotenv import load_dotenv
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.preprocessing import MinMaxScaler


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

def callback():
    st.session_state.button_clicked = True

@st.cache_data
def get_probability_percentage(api_url, client_information):
    """Infer the probability of loan payback from the client information
    :param client_information: The information of the given client in a JSON format
    """
    response = requests.post(api_url, json=client_information).json()
    return response


@st.cache_data
def search_client(api_url, sk_id):
    """Search for the client's information that's sent into the model.
    :param sk_id: The client ID for searching
    """
    request = {"sk_id": sk_id}
    response = requests.post(api_url, json=request).json()
    return response


@st.cache_data
def get_explainer_values(api_url, client_information):
    """Get the SHAP feature importance values"""
    response = requests.post(api_url, json=client_information).json()
    return response


@st.cache_data
def get_comparision_values(api_url):
    response = requests.post(api_url).json()
    return json.loads(response)


def change_df(dataframe):
    st.session_state.client_info_df = dataframe

def main():
    base_api_url = os.environ.get("API_URL")
    search_info_url = base_api_url + "/search"
    inference_url = base_api_url + "/infer"
    explainer_url = base_api_url + "/explain"
    comparision_url = base_api_url + "/compare"
    st.markdown(
        "<h1 style='text-align: center;'>Client loan repayability</h1>",
        unsafe_allow_html=True,
    )
    sk_id = st.text_input("Client ID:")
    if st.button("Search", on_click=callback) or st.session_state.button_clicked:
        if sk_id:
            client_info = search_client(search_info_url, int(sk_id))
        else:
            st.write("Please enter a valid client ID!")

        client_info_df = pd.DataFrame(json.loads(client_info))
        st.session_state.client_info_df = client_info_df
        client_info_df_edited = st.data_editor(client_info_df, use_container_width=True)
        answers = None
        if st.button("Submit"):
            left_column, right_column = st.columns(2)
            with left_column:
                answers = json.loads(get_probability_percentage(inference_url, json.loads(client_info_df_edited.to_json())))
                if answers["answer"] == 0:
                    st.write("### Client can repay his loan back!")
                else:
                    st.write("### Client cannot repay his loan back!")

                st.markdown("### How sure are we? (%)")
                fig = go.Figure(
                    go.Indicator(
                        domain={"x": [0, 1], "y": [0, 1]},
                        value=answers["answer_probability"] * 100,
                        mode="gauge+number+delta",
                        title={"text": "probability"},
                        delta={"reference": 50},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "steps": [
                                {"range": [0, 30], "color": "lightgreen"},
                                {"range": [30, 70], "color": "yellow"},
                                {"range": [70, 100], "color": "red"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": answers["answer_probability"] * 100,
                            },
                        },
                    )
                )
                st.plotly_chart(fig)
            with right_column:
                st.write("### Factors affecting the decision and their importance")
                shap_values = json.loads(get_explainer_values(explainer_url, json.loads(client_info_df_edited.to_json())))
                st.write("SHAP values summary plot")
                client_info_clean_df = client_info_df_edited.drop(columns=["SK_ID_CURR", "targets"], errors="ignore")
                fig2 = px.bar(
                    x=client_info_clean_df.columns,
                    y=shap_values["shap_values"],
                    title="Feature importance",
                )
                st.plotly_chart(fig2)

            st.write("---")

            title = ""
            if answers["answer"] == 0:
                title = "## How the client compares to people who cannot repay their loan"
            else:
                title = "## How the client compares to people who can repay their loan"

            st.markdown(title)
            st.markdown("### Client's economy vs global economy")

            comparison_res = get_comparision_values(comparision_url)
            comparision_df = pd.DataFrame(comparison_res)
            global_average = comparision_df.mean().tolist()
            local_average = (
                client_info_df.drop(columns=["targets"], errors="ignore")
                .mean()
                .tolist()
            )
            mm = MinMaxScaler()
            x_columns = comparision_df.columns
            means = pd.DataFrame(
                {
                    "global_average": global_average,
                    "client_average": local_average,
                }
            )
            means = pd.DataFrame(mm.fit_transform(means), columns=means.columns)
            means.index = x_columns
            required_indices = [
                "AMT_CREDIT",
                "AMT_INCOME_TOTAL",
                "EXT_SOURCE_2",
                "AMT_CREDIT_LIMIT_ACTUAL",
            ]
            means_short = means.loc[means.index.isin(required_indices)]
            fig3 = px.bar(
                means_short,
                x=means_short.index,
                y=["global_average", "client_average"],
                title=title,
                barmode="group",
            )
            for i, t in enumerate(["Rest of the clients", "Current client"]):
                fig3.data[i].text = t
                fig3.data[i].textposition = "outside"

            st.plotly_chart(fig3, use_container_width=True)

            # Assests and property
            st.write("---")
            st.markdown("### Client's housing and family situation vs global housing and family situation")
            left, right = st.columns(2)
            with left:
                owns_a_car = "Yes" if client_info_clean_df['FLAG_OWN_CAR_No'].tolist()[0] == 0 else "No"
                st.metric(label="Client owns a car?", value=owns_a_car)
                owns_cars_global = comparision_df["FLAG_OWN_CAR_No"].value_counts().to_frame()
                owns_cars_global = owns_cars_global.rename(columns={"FLAG_OWN_CAR_No": "Owns a car?"})
                owns_cars_global.index = ["No", "Yes"]
                fig4 = px.bar(owns_cars_global, x=owns_cars_global.index, y="count", title="All the clients who own a car")
                st.plotly_chart(fig4, use_container_width=True)

            with right:
                owns_realty = "Yes" if client_info_clean_df["FLAG_OWN_REALTY_Yes"].tolist()[0] == 1 else "No"
                st.metric(label="Client owns a property?", value=owns_realty)
                owns_realty_global = comparision_df["FLAG_OWN_REALTY_Yes"].value_counts().to_frame()
                owns_realty_global = owns_realty_global.rename(columns={"FLAG_OWN_REALTY_Yes": "Owns a property?"})
                owns_realty_global.index = ["Yes", "No"]
                fig5 = px.bar(owns_realty_global, x=owns_realty_global.index, y="count", title="All the clients that own a property.")
                st.plotly_chart(fig5)




if __name__ == "__main__":
    #load_dotenv()
    main()
