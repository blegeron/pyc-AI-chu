import json
import pandas as pd
import requests
import time
from datetime import datetime
from loguru import logger

# Configure loguru to write logs to a file and stdout
logger.add(
    "/home/user_jodert/code/joderthibaud/Personal/pyc-AI-chuvTHJ/logs/battle_scraper.log",
    rotation="1 day",  # Rotate logs every day
    retention="7 days",  # Keep logs for 7 days
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
    file_path = "/home/user_jodert/code/joderthibaud/Personal/pyc-AI-chuvTHJ/data/poke_battle.csv"
    data = fetch_battle_data()
    new_battles = process_battle_data(data)
    append_to_csv(new_battles, file_path)

if __name__ == "__main__":
    logger.info("Starting battle scraper...")
    try:
        while True:
            main()
            time.sleep(3600)  # Refresh every hour
    except KeyboardInterrupt:
        logger.info("Battle scraper stopped by user.")
