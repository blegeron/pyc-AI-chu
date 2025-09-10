import asyncio
import json
from typing import Any, Dict, Optional

import numpy as np
import ollama
from loguru import logger
from poke_env import AccountConfiguration, Player, RandomPlayer
from poke_env.environment.singles_env import Battle, Move, Pokemon

qwen_model = "qwen3:0.6b"


def normalize_name(name: str) -> str:
    """Lowercase and remove non-alphanumeric characters."""
    return "".join(filter(str.isalnum, name)).lower()


STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        "name": "choose_move",
        "description": "Selects and executes an available attacking or status move.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_name": {
                    "type": "string",
                    "description": "The exact name or ID (e.g., 'thunderbolt', 'swordsdance') of the move to use. Must be one of the available moves.",  # noqa: E501
                },
            },
            "required": ["move_name"],
        },
    },
    "choose_switch": {
        "name": "choose_switch",
        "description": "Selects an available Pokémon from the bench to switch into.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon_name": {
                    "type": "string",
                    "description": "The exact name of the Pokémon species to switch to (e.g., 'Pikachu', 'Charizard'). Must be one of the available switches.",  # noqa: E501
                },
            },
            "required": ["pokemon_name"],
        },
    },
}


class LLMAgentBase(Player):
    """Base class for LLM-based agents that can make decisions in battles."""

    def __init__(self, *args: any, **kwargs: any) -> None:
        """Initialize the agent with standard tools and an empty battle history."""
        super().__init__(*args, **kwargs)
        self.standard_tools = STANDARD_TOOL_SCHEMA
        self.battle_history = []

    def _format_battle_state(self, battle: Battle) -> str:
        active_pkmn = battle.active_pokemon
        active_pkmn_info = (
            f"Your active Pokemon: {active_pkmn.species} "
            f"(Type: {'/'.join(map(str, active_pkmn.types))}) "
            f"HP: {active_pkmn.current_hp_fraction * 100:.1f}% "
            f"Base stats: {active_pkmn.base_stats}% "
            f"Status: {active_pkmn.status.name if active_pkmn.status else 'None'} "
            f"Boosts: {active_pkmn.boosts}"
        )

        opponent_pkmn = battle.opponent_active_pokemon
        opp_info_str = "Unknown"
        if opponent_pkmn:
            opp_info_str = (
                f"{opponent_pkmn.species} "
                f"(Type: {'/'.join(map(str, opponent_pkmn.types))}) "
                f"HP: {opponent_pkmn.current_hp_fraction * 100:.1f}% "
                f"Base stats: {opponent_pkmn.base_stats}% "
                f"Status: {opponent_pkmn.status.name if opponent_pkmn.status else 'None'} "
                f"Boosts: {opponent_pkmn.boosts}"
            )
        opponent_pkmn_info = f"Opponent's active Pokemon: {opp_info_str}"

        available_moves_info = "Available moves:\n"
        if battle.available_moves:
            available_moves_info += "\n".join(
                [
                    f"- {move.id} (Type: {move.type}, BP: {move.base_power}, Acc: {move.accuracy}, PP: {
                        move.current_pp
                    }/{move.max_pp}, Cat: {move.category.name})"
                    for move in battle.available_moves
                ]
            )
        else:
            available_moves_info += "- None (Must switch or Struggle)"

        available_switches_info = "Available switches:\n"
        if battle.available_switches:
            available_switches_info += "\n".join(
                [
                    f"- {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%, Status: {
                        pkmn.status.name if pkmn.status else 'None'
                    })"
                    for pkmn in battle.available_switches
                ]
            )
        else:
            available_switches_info += "- None"

        state_str = (
            f"{active_pkmn_info}\n"
            f"{opponent_pkmn_info}\n\n"
            f"{available_moves_info}\n\n"
            f"{available_switches_info}\n\n"
            f"Weather: {battle.weather}\n"
            f"Terrains: {battle.fields}\n"
            f"Your Side Conditions: {battle.side_conditions}\n"
            f"Opponent Side Conditions: {battle.opponent_side_conditions}"
        )
        return state_str.strip()

    def _find_move_by_name(self, battle: Battle, move_name: str) -> Optional[Move]:
        normalized_name = normalize_name(move_name)
        # Prioritize exact ID match
        for move in battle.available_moves:
            if move.id == normalized_name:
                return move
        # Fallback: Check display name (less reliable)
        for move in battle.available_moves:
            if move.name.lower() == move_name.lower():
                logger.warning(
                    f"Matched move by display name '{move.name}' instead of ID '{move.id}'. Input was '{move_name}'."
                )
                return move
        return None

    def _find_pokemon_by_name(self, battle: Battle, pokemon_name: str) -> Optional[Pokemon]:
        normalized_name = normalize_name(pokemon_name)
        for pkmn in battle.available_switches:
            # Normalize the species name for comparison
            if normalize_name(pkmn.species) == normalized_name:
                return pkmn
        return None

    async def choose_move(self, battle: Battle) -> str:
        """Make a decision based on the battle state using an LLM."""
        battle_state_str = self._format_battle_state(battle)
        decision_result = await self._get_llm_decision(battle_state_str)
        logger.info(decision_result)
        decision = decision_result.get("decision")
        error_message = decision_result.get("error")
        action_taken = False
        fallback_reason = ""

        if decision:
            function_name = decision.get("name")
            args = decision.get("arguments", {})
            if function_name == "choose_move":
                move_name = args.get("move_name")
                if move_name:
                    chosen_move = self._find_move_by_name(battle, move_name)
                    if chosen_move and chosen_move in battle.available_moves:
                        action_taken = True
                        chat_msg = f"AI Decision: Using move '{chosen_move.id}'."
                        logger.info(chat_msg)
                        return self.create_order(chosen_move)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid move '{move_name}'."
                else:
                    fallback_reason = "LLM 'choose_move' called without 'move_name'."
            elif function_name == "choose_switch":
                pokemon_name = args.get("pokemon_name")
                if pokemon_name:
                    chosen_switch = self._find_pokemon_by_name(battle, pokemon_name)
                    if chosen_switch and chosen_switch in battle.available_switches:
                        action_taken = True
                        chat_msg = f"AI Decision: Switching to '{chosen_switch.species}'."
                        logger.info(chat_msg)
                        return self.create_order(chosen_switch)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid switch '{pokemon_name}'."
                else:
                    fallback_reason = "LLM 'choose_switch' called without 'pokemon_name'."
            else:
                fallback_reason = f"LLM called unknown function '{function_name}'."

        if not action_taken:
            if not fallback_reason:
                if error_message:
                    fallback_reason = f"API Error: {error_message}"
                elif decision is None:
                    fallback_reason = "LLM did not provide a valid function call."
                else:
                    fallback_reason = "Unknown error processing LLM decision."

            logger.warning(f"{fallback_reason} Choosing random action.")

            if battle.available_moves or battle.available_switches:
                return self.choose_random_move(battle)
            else:
                logger.info("AI Fallback: No moves or switches available. Using Struggle/Default.")
                return self.choose_default_move(battle)

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _get_llm_decision")


class QwenAgent(LLMAgentBase):
    """LLM-based agent that uses the Qwen model for decision making."""

    def __init__(self, model: str = qwen_model, avatar: str = "rosa", *args: any, **kwargs: any):
        kwargs["avatar"] = avatar
        kwargs["start_timer_on_battle_start"] = True
        super().__init__(*args, **kwargs)
        self.model = model
        self.qwen_tools = []
        for _, tool_schema in self.standard_tools.items():
            self.qwen_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_schema["name"],
                        "description": tool_schema["description"],
                        "parameters": tool_schema["parameters"],
                    },
                }
            )

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        # Placeholder for actual LLM call
        # This should return a dict with 'decision' or 'error' keys
        # Example response structure:
        system_prompt = (
            "You are a skilled Pokemon battle AI. Your goal is to win the battle. "
            "Based on the current battle state, decide the best action: either use an available move or switch to an available Pokémon. "
            "Consider type matchups, HP, base stats, status conditions, field effects, entry hazards, and potential opponent actions. "
            "Only choose actions listed as available using their exact ID (for moves) or species name (for switches). "
            "Use the provided functions to indicate your choice."
        )
        user_prompt = f"Current Battle State:\n{battle_state}\n\nChoose the best action by calling the appropriate function ('choose_move' or 'choose_switch')."

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=self.qwen_tools,
            )
            logger.info(f"Qwen RESPONSE : {response.message.content}")
            logger.info(f"Tool calls : {response.message.tool_calls}")
            # Check for tool calls in the response
            if response.message.tool_calls:
                tool_call = response.message.tool_calls[0]  # Get the first tool call
                function_name = tool_call.function.name
                logger.info(f"Qwen called function: {function_name} with arguments: {tool_call.function.arguments}")
                try:
                    arguments = tool_call.function.arguments
                    if function_name in self.standard_tools:
                        return {"decision": {"name": function_name, "arguments": arguments}}
                    else:
                        return {"error": f"Model called unknown function '{function_name}'."}
                except json.JSONDecodeError:
                    return {"error": f"Error decoding function arguments: {tool_call.function.arguments}"}
            else:
                # Model decided not to call a function
                return {"error": f"Qwen did not return a function call. Response: {response.content}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e!s}"}


if __name__ == "__main__":
    qwen_player = QwenAgent()

    account_configuration = AccountConfiguration(
        username=f"RANDOM{np.random.randint(1e5)}",
        password=None,
    )
    second_player = RandomPlayer(account_configuration=account_configuration)

    asyncio.run(qwen_player.battle_against(second_player, n_battles=1))
