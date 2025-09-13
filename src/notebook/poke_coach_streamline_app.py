import streamlit as st
import subprocess
import os

def main():
    st.title('My Pokemon coach')
    st.write('Here are the best Pokemons to use !')
    st.markdown("*Further resources [here](https://altair-viz.github.io/gallery/selection_histogram.html)*")
    
# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Function to execute a script
def execute_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    subprocess.run(["python", script_path], check=True)

# Streamlit UI
st.title("Pokemon Data Processing")

if st.button("Click to launch"):
    with st.spinner("Running scripts..."):
        try:
            execute_script("src/notebook/scrapping_battles.py")
            execute_script("src/notebook/scrapping_battles_poke.py")
            execute_script("src/notebook/detailed_poke_turns.py")
            st.success("Scripts executed successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"Error executing scripts: {e}")


if __name__ == "__main__":
    main()