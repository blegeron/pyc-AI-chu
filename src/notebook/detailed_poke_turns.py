#!/usr/bin/env python3
import requests
import pandas as pd
import re
import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "../../logs")
DATA_DIR = os.path.join(BASE_DIR, "../../data")
POKE_BATTLE_FILE = os.path.join(DATA_DIR, "poke_battle.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "battle_details.csv")

def run_detailed_poke_turns(DATA_DIR, LOGS_DIR):

    def parse_log_with_block_entry_hp(lines, battle_ref):
        current_ts = None
        poke_p1 = None
        poke_p2 = None
        hp_p1 = 100
        hp_p2 = 100
        pre_hp_p1 = None
        pre_hp_p2 = None
        block_entry_hp_p1 = None
        block_entry_hp_p2 = None
        action_p1 = ""
        action_p2 = ""
        victory_flag = 0
        rows = []

        def append_row_for_ts(ts):
            nonlocal victory_flag
            if not (poke_p1 and poke_p2):
                return
            ori1 = block_entry_hp_p1 if block_entry_hp_p1 is not None else (pre_hp_p1 if pre_hp_p1 is not None else hp_p1)
            ori2 = block_entry_hp_p2 if block_entry_hp_p2 is not None else (pre_hp_p2 if pre_hp_p2 is not None else hp_p2)
            diff = (hp_p1 - ori1) - (hp_p2 - ori2)
            rows.append({
                "Poke_battle_id": battle_ref,
                "Turn_id": ts,
                "Poke_p1": poke_p1,
                "Poke_p2": poke_p2,
                "PokemonP1_HP_ORI": int(ori1),
                "PokemonP2_HP_ORI": int(ori2),
                "Action_P1": action_p1,
                "Action_P2": action_p2,
                "PokemonP1_HP": int(hp_p1),
                "PokemonP2_HP": int(hp_p2),
                "Difference": int(diff),
                "Move": 0,
                "Victory": victory_flag
            })
            # reset for next turn
            victory_flag = 0

        for line in lines:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            evt = parts[1]

            if evt == "t:":
                new_ts = parts[2] if len(parts) > 2 else None
                if current_ts is not None:
                    append_row_for_ts(current_ts)
                current_ts = new_ts
                pre_hp_p1 = hp_p1
                pre_hp_p2 = hp_p2
                block_entry_hp_p1 = None
                block_entry_hp_p2 = None
                action_p1 = ""
                action_p2 = ""
                continue

            if evt == "switch":
                player = parts[2]
                pokemon = parts[3].split(",")[0] if len(parts) > 3 else None
                hp_str = parts[4] if len(parts) > 4 else None
                if player.startswith("p1"):
                    poke_p1 = pokemon
                    if hp_str and "/" in hp_str:
                        try: hp_p1 = int(hp_str.split("/")[0])
                        except: pass
                    if block_entry_hp_p1 is None:
                        block_entry_hp_p1 = hp_p1
                    action_p1 = "Switch"
                elif player.startswith("p2"):
                    poke_p2 = pokemon
                    if hp_str and "/" in hp_str:
                        try: hp_p2 = int(hp_str.split("/")[0])
                        except: pass
                    if block_entry_hp_p2 is None:
                        block_entry_hp_p2 = hp_p2
                    action_p2 = "Switch"
                continue

            if evt == "drag":
                player = parts[2]
                pokemon = parts[3].split(",")[0] if len(parts) > 3 else None
                hp_str = parts[4] if len(parts) > 4 else None
                if player.startswith("p1"):
                    poke_p1 = pokemon
                    if hp_str and "/" in hp_str:
                        try: hp_p1 = int(hp_str.split("/")[0])
                        except: pass
                    if block_entry_hp_p1 is None:
                        block_entry_hp_p1 = hp_p1
                elif player.startswith("p2"):
                    poke_p2 = pokemon
                    if hp_str and "/" in hp_str:
                        try: hp_p2 = int(hp_str.split("/")[0])
                        except: pass
                    if block_entry_hp_p2 is None:
                        block_entry_hp_p2 = hp_p2
                continue

            if evt == "move":
                player = parts[2]
                move = parts[3] if len(parts) > 3 else ""
                if player.startswith("p1"): action_p1 = move
                elif player.startswith("p2"): action_p2 = move
                continue

            if evt == "cant":
                player = parts[2]
                if player.startswith("p1"): action_p1 = "cant"
                elif player.startswith("p2"): action_p2 = "cant"
                continue

            if evt == "faint":
                player = parts[2]
                if player.startswith("p1"):
                    victory_flag = -1
                elif player.startswith("p2"):
                    victory_flag = 1
                continue

            if evt in ("-damage", "-heal"):
                player = parts[2]
                hp_token = parts[3] if len(parts) > 3 else ""
                match = re.search(r"(\d+)(?:/(\d+))?", hp_token)
                if match:
                    hp_val = int(match.group(1))
                    if player.startswith("p1"): hp_p1 = hp_val
                    elif player.startswith("p2"): hp_p2 = hp_val
                continue

        if current_ts is not None:
            append_row_for_ts(current_ts)

        return rows


    def compute_move(df):
        for i in range(len(df) - 1):
            cur, nxt = df.loc[i], df.loc[i+1]
            mv = 0
            if str(cur["Action_P1"]) in ("Switch",""): mv += 1
            if str(cur["Action_P2"]) in ("Switch",""): mv -= 1
            if str(nxt["Action_P1"]) in ("Switch",""): mv -= 1
            if str(nxt["Action_P2"]) in ("Switch",""): mv += 1
            df.at[i,"Move"] = mv
        return df


    def main():
        battles = pd.read_csv(POKE_BATTLE_FILE)
        all_rows = []

        for _, row in battles.iterrows():
            battle_ref = row["battle_ref"]
            url = f"https://replay.pokemonshowdown.com/{battle_ref}.json"
            print(f"Fetching {url} ...")
            try:
                response = requests.get(url)
                data = response.json()
                log = data.get("log","")
                lines = log.splitlines()
                rows = parse_log_with_block_entry_hp(lines, battle_ref)
                if rows:
                    df = pd.DataFrame(rows)
                    df = compute_move(df)
                    all_rows.append(df)
            except Exception as e:
                print(f"Failed to fetch/parse {battle_ref}: {e}")

        if all_rows:
            final_df = pd.concat(all_rows, ignore_index=True)
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved details for {len(all_rows)} battles to {OUTPUT_FILE}")
        else:
            print("No battles parsed.")

    def generate_poke_moves():
        details_file = os.path.join(DATA_DIR, "battle_details.csv")
        output_file = os.path.join(DATA_DIR, "poke_moves.csv")

        if not os.path.exists(details_file):
            print("battle_details.csv not found, run main() first.")
            return

        df = pd.read_csv(details_file)
        rows = []

        for _, row in df.iterrows():
            # Player 1 perspective
            rows.append({
                "Poke_battle_id": row["Poke_battle_id"],
                "Turn_id": row["Turn_id"],
                "Poke": row["Poke_p1"],
                "Poke_opponent": row["Poke_p2"],
                "Action_Poke": row["Action_P1"],
                "Action_Poke_opponent": row["Action_P2"],
                "Difference": row["Difference"],
                "Move": row["Move"],
                "Victory": row["Victory"]
            })

            # Player 2 perspective (flip values)
            rows.append({
                "Poke_battle_id": row["Poke_battle_id"],
                "Turn_id": row["Turn_id"],
                "Poke": row["Poke_p2"],
                "Poke_opponent": row["Poke_p1"],
                "Action_Poke": row["Action_P2"],
                "Action_Poke_opponent": row["Action_P1"],
                "Difference": -row["Difference"],
                "Move": -row["Move"],
                "Victory": -row["Victory"]
            })

        moves_df = pd.DataFrame(rows)
        moves_df.to_csv(output_file, index=False)
        print(f"Saved poke_moves.csv with {len(moves_df)} rows to {output_file}")



    if __name__ == "__main__":
        main()
        generate_poke_moves()
