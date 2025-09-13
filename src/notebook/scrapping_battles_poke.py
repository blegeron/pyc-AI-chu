import pandas as pd
import requests
from loguru import logger
import uuid
import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LOGS_DIR = os.path.join(BASE_DIR, "../../logs")


def run_scrapping_battles_poke(DATA_DIR, LOGS_DIR):

    # Load the existing CSV file
    input_csv_path = os.path.join(DATA_DIR, "poke_battle.csv") 
    output_csv_path = os.path.join(DATA_DIR, "poke_battle_details.csv") 

    # Read the input CSV file
    df = pd.read_csv(input_csv_path)

    def fetch_battle_details(battle_ref, player_p1, player_p2):
        url = f"https://replay.pokemonshowdown.com/{battle_ref}.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Extract log
            log = data.get("log", "")

            # Extract gen and tier
            gen, tier = None, None
            for line in log.split('\n'):
                if "|gen|" in line:
                    gen = line.split("|gen|")[1].split("|")[0]
                if "|tier|" in line:
                    tier = line.split("|tier|")[1].split("|")[0]

            # Extract winner
            winner = None
            for line in log.split('\n'):
                if line.startswith("|win|"):
                    winner = line.split("|win|")[1]

            # Extract Pokémon for both players
            poke_details = []
            for line in log.split('\n'):
                if line.startswith("|poke|p1|"):
                    poke_battle_ref = line.split("|poke|p1|")[1].split(",")[0]
                    unique_id = str(uuid.uuid4())
                    poke_details.append({
                        "unique_id": unique_id,
                        "battle_ref": battle_ref,
                        "poke_battle_ref": poke_battle_ref,
                        "player": player_p1,
                        "gen": gen,
                        "tier": tier,
                        "winner": 1 if player_p1 == winner else 0
                    })
                elif line.startswith("|poke|p2|"):
                    poke_battle_ref = line.split("|poke|p2|")[1].split(",")[0]
                    unique_id = str(uuid.uuid4())
                    poke_details.append({
                        "unique_id": unique_id,
                        "battle_ref": battle_ref,
                        "poke_battle_ref": poke_battle_ref,
                        "player": player_p2,
                        "gen": gen,
                        "tier": tier,
                        "winner": 1 if player_p2 == winner else 0
                    })

            return poke_details
        except Exception as e:
            logger.error(f"Failed to fetch or parse data for {battle_ref}: {e}")
            return []

    def append_to_csv(new_data, file_path):
        if not new_data:
            logger.warning("No new data to append.")
            return
        try:
            try:
                existing_data = pd.read_csv(file_path)
            except FileNotFoundError:
                existing_data = pd.DataFrame()

            df = pd.DataFrame(new_data)
            updated_data = pd.concat([existing_data, df], ignore_index=True)
            updated_data.drop_duplicates(subset="unique_id", keep="first", inplace=True)
            updated_data.to_csv(file_path, index=False)
            logger.info(f"Successfully appended {len(new_data)} new entries to {file_path}")
        except Exception as e:
            logger.error(f"Failed to append data to CSV: {e}")

    # Fetch details for each battle_ref
    battle_details = []
    for _, row in df.iterrows():
        battle_ref = row["battle_ref"]
        player_p1 = row["Player1"]
        player_p2 = row["Player2"]
        details = fetch_battle_details(battle_ref, player_p1, player_p2)
        battle_details.extend(details)

    # Append or create the CSV file
    append_to_csv(battle_details, output_csv_path)
