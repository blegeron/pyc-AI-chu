import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL")


@st.cache_data
def load_data():
    return requests.get(f"{BACKEND_URL}/agents").json()


agent = load_data()
st.button("Hello")

st.header("Pokemon AI Duel")
st.write(
    """ Try out our AI-powered Pokemon duel simulator! Enter your pseudo and choose an AI agent to battle against. Watch as the AI makes strategic moves in real-time. """
)

pseudo = st.text_input("Enter your Pseudo", value="AshKetchum")
agent = st.selectbox("Choose an AI Agent", options=agent.get("agents", []))

if st.button("Predict Species"):
    with st.spinner("Predicting species..."):
        time.sleep(3)
        params = {
            "pseudo": pseudo,
            "agent": agent,
        }

        response = requests.get(f"{BACKEND_URL}/duel", params=params)

    if response.status_code == 200:
        st.snow()
        prediction = response.json()
        st.success(f"The predicted species is: {prediction}")
    else:
        st.error("Error occurred while predicting species.")

st.header("Team Generator")
st.write(""" Generate a competitive Pokemon team based on your preferred format and constraints. Whether you're looking
for a balanced team or one that fits specific criteria, our generator has you covered! """)

format = st.text_input("Enter Format", value="gen9ou")
constraint = st.text_input("Enter Constraint (optional)", value="")

if st.button("Generate Team"):
    with st.spinner("Generating team..."):
        time.sleep(2)
        params = {
            "format": format,
            "constraint": constraint if constraint else None,
        }

        response = requests.get(f"{BACKEND_URL}/team", params=params)

    if response.status_code == 200:
        team = response.json().get("team", "")
        st.success("Generated Team:")
        st.code(team)
    else:
        st.error("Error occurred while generating team.")
