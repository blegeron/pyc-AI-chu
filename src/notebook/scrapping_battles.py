import json
import pandas as pd
import requests
import time
import os
from datetime import datetime
from loguru import logger

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LOGS_DIR = os.path.join(BASE_DIR, "../../logs")

def run_scrapping_battles(DATA_DIR, LOGS_DIR):
    # Define paths relative to data_dir and logs_dir
    file_path = os.path.join(DATA_DIR, "poke_battle.csv")

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Configure loguru to write logs to a file and stdout
    log_file_path = os.path.join(LOGS_DIR, "battle_scraper.log")
    logger.add(
        os.path.join(LOGS_DIR, "battle_scraper.log"),
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    def fetch_battle_data():
        url = "https://replay.pokemonshowdown.com/search.json?format=gen9ou"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data: {e}")
            return None

    def process_battle_data(data):
        if not data:
            logger.warning("No data to process.")
            return []
        return [
            {
                "Player1": battle["players"][0],
                "Player2": battle["players"][1],
                "battle_ref": battle["id"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            for battle in data
        ]

    def append_to_csv(new_data, file_path):
        if not new_data:
            logger.warning("No new data to append.")
            return
        try:
            try:
                existing_data = pd.read_csv(file_path)
                df = pd.DataFrame(new_data)
                updated_data = pd.concat([existing_data, df], ignore_index=True)
                updated_data.drop_duplicates(subset="battle_ref", keep="first", inplace=True)
            except FileNotFoundError:
                updated_data = pd.DataFrame(new_data)
            updated_data.to_csv(file_path, index=False)
            logger.info(f"Successfully appended {len(new_data)} new battles to {file_path}")
        except Exception as e:
            logger.error(f"Failed to append data to CSV: {e}")

    def main():
        file_path = os.path.join(DATA_DIR, "poke_battle.csv")
        data = fetch_battle_data()
        new_battles = process_battle_data(data)
        append_to_csv(new_battles, file_path)

    logger.info("Starting battle scraper...")
    main()
    logger.info("Battle scraper completed.")