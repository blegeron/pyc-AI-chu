import streamlit as st
import subprocess
import os
import sys


def read_last_log_entry(logs_dir, log_file_name):
    log_file_path = os.path.join(logs_dir, log_file_name)
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()
            if lines:
                # Display the last log entries or fewer if there aren't enough
                return "".join(lines[-2:])
    return "No log entries found."


def main():
    st.title("My Pokemon coach")
    st.write("Here are the best Pokemons to use !")
    st.markdown("*Further resources [here](https://altair-viz.github.io/gallery/selection_histogram.html)*")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    st.title("Pokemon Data Processing")

    if st.button("Click to load"):
        with st.spinner("Running scripts..."):
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), "src", "notebook"))
                from detailed_poke_turns import run_detailed_poke_turns
                from scrapping_battles import run_scrapping_battles
                from scrapping_battles_poke import run_scrapping_battles_poke

                run_scrapping_battles(DATA_DIR, LOGS_DIR)
                st.subheader("Scrapping Battles Log:")
                battles_log = read_last_log_entry(LOGS_DIR, "battle_scraper.log")
                st.text_area("Last Log Entries for scrapping_battles:", battles_log, height=150)

                run_scrapping_battles_poke(DATA_DIR, LOGS_DIR)
                st.subheader("Scrapping Battles Poke Log:")
                battles_poke_log = read_last_log_entry(
                    LOGS_DIR, "battle_poke_scraper.log"
                )  # Update with correct log file name
                st.text_area("Last Log Entries for scrapping_battles_poke:", battles_poke_log, height=150)

                run_detailed_poke_turns(DATA_DIR, LOGS_DIR)
                st.subheader("Detailed Poke Turns Log:")
                poke_turns_log = read_last_log_entry(LOGS_DIR, "poke_turns.log")  # Update with correct log file name
                st.text_area("Last Log Entries for detailed_poke_turns:", poke_turns_log, height=150)

                # run_scrapping_battles(DATA_DIR, LOGS_DIR)
                # # Display the last log entries
                # last_log_entries = read_last_log_entry(LOGS_DIR, "battle_scraper.log")
                # st.text_area("Last Log Entries:", last_log_entries, height=200)
            except Exception as e:
                st.error(f"Error executing script: {e}")


if __name__ == "__main__":
    main()
