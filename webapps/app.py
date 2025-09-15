import os
import time

import requests
import streamlit as st

BACKEND_URI = f"{os.getenv('BACKEND_URL')}:{os.getenv('API_PORT', 8888)}"


@st.cache_data
def load_data():
    with st.spinner("Loading data..."):
        data = requests.get(f"{BACKEND_URI}/agents")
    return data.json() if data.status_code == 200 else {}


agents = load_data()
st.write(f"{agents}")

st.header("Pokemon AI Duel")
st.write(
    """ Try out our AI-powered Pokemon duel simulator! Enter your pseudo and choose an AI agent to battle against. Watch as the AI makes strategic moves in real-time. """
)

pseudo = st.text_input("Enter your Pseudo", value="AshKetchum")
agent = st.selectbox("Choose an AI Agent", options=[*agents.get("agents", []), "pokemon_agentic"])
format = st.selectbox("Choose a format", options=["gen9randombattle", "gen9ou"])

if st.button("Let's duel!!!"):
    with st.spinner("Duel is declared"):
        params = {"pseudo": pseudo, "agent": agent, "format": format}

        response = requests.get(f"{BACKEND_URI}/duel", params=params)

    if response.status_code == 200:
        st.success("The duel is finished!")
    else:
        st.error("Error occurred while demanding the duel.")

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

        response = requests.get(f"{BACKEND_URI}/team", params=params)

    if response.status_code == 200:
        team = response.json().get("team", "")
        st.success("Generated Team:")
        st.code(team)
    else:
        st.error("Error occurred while generating team.")
