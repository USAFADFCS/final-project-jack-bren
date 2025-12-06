# Agent generated from PerplexityAI
# C1C Jack West and Brendan Wu
# CS471 Fall 2025
# Documentation: Included in separate file in the repository
# Guide on how to run the program effectively is in the README word file
import os
import asyncio
import requests
import re
from fairlib import HuggingFaceAdapter, ToolRegistry, ToolExecutor, WorkingMemory, ReActPlanner, SimpleAgent, RoleDefinition
from huggingface_hub import HfFolder



# Base class for FAIR-LLM compatible tool.
class Tool:
    
    def get_tool_name(self):
        raise NotImplementedError("Tool must define tool name.")

    def get_tool_description(self):
        raise NotImplementedError("Tool must define tool description.")

    def call(self, args):
        """Main entrypoint for tool. Accepts a dictionary or string of arguments."""
        raise NotImplementedError("Tool must implement 'call' method.")

# Tool allows agent to find recall information for given vehicle
class RecallLookupTool(Tool):
    def __init__(self):
        self.name = self.get_tool_name()
        self.description = self.get_tool_description()

    def get_tool_name(self):
        return "RecallLookup"


    def get_tool_description(self):
        return "Fetches NHTSA recall summaries for a specified year/make/model."

    def call(self, args):
        # Parses make, model, and year from user input
        make, model, year = parse_vehicle_info(args['query'])
        if not all([make, model, year]):
            return "Could not parse vehicle info from input."
        # Separate function makes the API call 
        return fetch_nhtsa_recalls(make, model, year)

# Tool allows agent to find safety ratings for given vehicle
class RatingsLookupTool(Tool):
    def __init__(self):
        self.name = self.get_tool_name()
        self.description = self.get_tool_description()

    def get_tool_name(self):
        return "RatingsLookup"

    def get_tool_description(self):
        return "Fetches NHTSA 5‑Star safety ratings for a specified year/make/model."

    def call(self, args):
        make, model, year = parse_vehicle_info(args["query"])
        if not all([make, model, year]):
            return "Could not parse vehicle info from input."

        # No separate fetch function

        # First API call to find the NHTSA vehicle ID from make, model, and year
        search_url = (
            f"https://api.nhtsa.gov/SafetyRatings/modelyear/{year}"
            f"/make/{make}/model/{model}"
        )
        try:
            resp = requests.get(search_url, timeout=10)
            resp.raise_for_status()
            variants = resp.json().get("Results", []) or resp.json().get("results", [])
            if not variants:
                return f"No safety rating variants found for {year} {make} {model}."

            # Take the first 1–2 variants only to keep output short
            summaries = []
            for v in variants[:1]:
                vehicle_id = v.get("VehicleId") or v.get("VehicleId".lower())
                desc = v.get("VehicleDescription", "Unknown variant")
                if not vehicle_id:
                    continue

                # Take the found vehicle ID, make another API call to fetch safety ratings
                rating_url = f"https://api.nhtsa.gov/SafetyRatings/VehicleId/{vehicle_id}"
                r2 = requests.get(rating_url, timeout=10)
                r2.raise_for_status()
                rating_results = r2.json().get("Results", []) or r2.json().get("results", [])
                if not rating_results:
                    continue

                r = rating_results[0]
                overall = r.get("OverallRating", "Not Rated")
                front = r.get("OverallFrontCrashRating", "Not Rated")
                side = r.get("OverallSideCrashRating", "Not Rated")
                rollover = r.get("RolloverRating", "Not Rated")

                # Summary of found safety rating information for the LLM to consider in response
                summaries.append(
                    f"- {desc}: overall {overall}, front {front}, side {side}, rollover {rollover}"
                )

            if not summaries:
                return f"No detailed ratings found for {year} {make} {model}."

            return (
                f"NHTSA safety ratings for {year} {make} {model} (top variants):\n"
                + "\n".join(summaries)
            )

        except Exception as e:
            return f"Error fetching ratings: {str(e)}"



import re

# List of popular car makes for the parser to find within user input
KNOWN_MAKES = [
    "Toyota", "Honda", "Ford", "Nissan", "Chevrolet", "BMW", "Mercedes", "Volkswagen", "Hyundai", "Kia",
    "Mazda", "Subaru", "Audi", "Jeep", "Dodge", "Ram", "GMC", "Lexus", "Acura", "Infiniti", "Tesla",
    "Volvo", "Porsche", "Jaguar", "Mitsubishi", "Buick", "Cadillac", "Chrysler", "Land Rover"
]

# parser function; identifies make, model, and year from user input
def parse_vehicle_info(user_input):

    # Finds 4-digit year between 1980-2099 (regex)
    year_match = re.search(r"\b(19|20)\d{2}\b", user_input)
    if not year_match:
        return None, None, None
    year = year_match.group()

    # Find car make (case insensitive)
    make = None
    for known_make in KNOWN_MAKES:
        pattern = r'\b' + re.escape(known_make) + r'\b'
        if re.search(pattern, user_input, re.IGNORECASE):
            make = known_make
            break

    if not make:
        return None, None, None

    # Find car model by seeking words after year and make
    # after year, expect make somewhere in following tokens
    tokens = user_input.split()
    try:
        year_index = tokens.index(year)
    except ValueError:
        year_index = -1

    if year_index == -1:
        return None, None, None

    # Look for make after year
    model_tokens = []
    found_make = False
    for token in tokens[year_index + 1:]:
        if not found_make and token.lower() == make.lower():
            found_make = True
            continue
        if found_make:
            model_tokens.append(token)
    model = " ".join(model_tokens).strip()
    return make, model, year


# Recall information function (called in RecallLookupTool())
def fetch_nhtsa_recalls(make, model, year):

    # API call to NHTSA recall database
    url = f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        recalls = r.json().get("results", [])
        if not recalls:
            return f"No recalls found for {year} {make} {model}."

        # Takes the first 3 recalls only to save space
        # Summaries information from JSON for LLM to use in response
        summary_list = []
        for recall in recalls[:3]:  
            component = recall.get("Component", "Unknown")
            rtype = recall.get("RecallType", "N/A")
            campaign = recall.get("NHTSACampaignNumber", "N/A")
            
            summary_list.append(
                f"- {component} (type: {rtype}, campaign: {campaign})"
            )

        return f"Top recalls for {year} {make} {model}:\n" + "\n".join(summary_list)
    except Exception as e:
        return f"Error fetching recalls: {str(e)}"


# Creation of FairLLM Agent
def create_agent(model_name, role_prompt, tools=None, memory=None):
   
    # Chosen model (from HuggingFace)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    
    # ---- IMPORTANT ----
    # The program did not run unless we hard-coded our HuggingFace token in here. Token is removed to allow file to be pushed to GitHub
    llm = HuggingFaceAdapter(model_name) #auth_token="copy/paste from in word doc")

    # Register tools
    tool_registry = ToolRegistry()
    if tools:
        for tool in tools:
            tool_registry.register_tool(tool)
    executor = ToolExecutor(tool_registry)
    working_memory = memory if memory else WorkingMemory()
    planner = ReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(role_prompt)
    agent = SimpleAgent(
        llm=llm, planner=planner, tool_executor=executor, memory=working_memory, max_steps=3
    )
    return agent

# Main program loop
async def main():

    # Define tools
    recall_tool = RecallLookupTool()
    ratings_tool = RatingsLookupTool()
    tools = [recall_tool, ratings_tool]

    # Explain role of the agent
    role_prompt = (
        "You are an expert auto safety and reliability assistant talking to an uninformed car buyer who does not know a lot about vehicles. "
    )

    # Instantiate the agent
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    agent = create_agent(model_name, role_prompt, tools=tools, memory=None)

    # Prompt the user for input
    print("Hello, I am a car-buying assistant. Type a question about a vehicle (e.g. '2018 Toyota Camry'). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Agent: Goodbye!")
                break

            # Call our tools with the user's query as input
            recall_info = recall_tool.call({"query": user_input})
            ratings_info = ratings_tool.call({"query": user_input})
            # Display results of API call returns to the user
            print("Recall info:", recall_info)
            print("Ratings info:", ratings_info)

            # Prompt for the LLM agent 
            prompt = (
                "Give a simple, non-technical summary of this vehicle's repair concerns, ratings, and trims and features for a used-car buyer"
                f"Recalls:\n{recall_info}\n\nRatings:\n{ratings_info}\n"
            )
            
            # Response of the LLM summary displayed to the user
            llm_response = await agent.arun(prompt)
            print("Agent summary:", llm_response)
        except Exception as e:
            print("Error:", e)
        except KeyboardInterrupt:
            print("Exiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())