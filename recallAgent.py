# Agent generated from PerplexityAI


import os
import asyncio
import requests
import re
from fairlib import HuggingFaceAdapter, ToolRegistry, ToolExecutor, WorkingMemory, ReActPlanner, SimpleAgent, RoleDefinition

# ----- Helper: Parse Make, Model, Year -----
def parse_vehicle_info(user_input):
    match = re.match(r"(?:what.*)?(\\d{4})\\s+([a-zA-Z]+)\\s+([\\w\\s\\-]+)", user_input, re.IGNORECASE)
    if match:
        year, make, model = match.groups()
        return make.strip(), model.strip(), year.strip()
    tokens = user_input.strip().split()
    year = None
    for token in tokens:
        if token.isdigit() and 1900 < int(token) < 2100:
            year = token
            break
    if year:
        idx = tokens.index(year)
        if len(tokens) > idx + 2:
            make = tokens[idx+1]
            model = " ".join(tokens[idx+2:])
            return make, model, year
    return None, None, None

# ----- Retrieval function -----
def fetch_nhtsa_recalls(make, model, year):
    print(make)
    print(model)
    print(year)

    url = f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        recalls = r.json().get("results", [])
        if not recalls:
            return f"No recalls found for {year} {make} {model}."
        summary_list = []
        for recall in recalls[:5]:
            summary_list.append(
                f"- {recall.get('Component','Unknown')} ({recall.get('RecallType','N/A')}): {recall.get('Summary','No summary')}"
            )
        return f"Top recalls for {year} {make} {model}:\n" + "\n".join(summary_list)
    except Exception as e:
        return f"Error fetching recalls: {str(e)}"

# ----- FairLLM agent creation -----
def create_agent(model_name, role_prompt, tools=None, memory=None):
    llm = HuggingFaceAdapter(model_name)
    tool_registry = ToolRegistry()
    if tools:
        for tool in tools:
            tool_registry.register_tool(tool)
    executor = ToolExecutor(tool_registry)
    working_memory = memory if memory else WorkingMemory()
    planner = ReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(role_prompt)
    agent = SimpleAgent(
        llm=llm, planner=planner, tool_executor=executor, memory=working_memory, max_steps=10
    )
    return agent

# ----- Main event loop -----
async def main():
    role_prompt = (
        "You are an expert auto safety and reliability assistant talking to an uninformed car buyer who does not know a lot about vehicles. "
        "Interpret NHTSA recall data and clearly summarize key risks or actions that the buyer should know about the vehicle's recall history so that they can make an informed decision on whether to buy the vehicle or not."
    )
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    agent = create_agent(model_name, role_prompt)

    print("Agent initialized. Type a question about a vehicle (e.g. '2018 Toyota Camry'). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Agent: Goodbye!")
                break

            # ---- RAG Pipeline: fetch recall info ----
            make, model, year = parse_vehicle_info(user_input)
            if make and model and year:
                recall_info = fetch_nhtsa_recalls(make, model, year)
                print("Raw recall info fetched from NHTSA:\n", recall_info)
                # --- Summarize with LLM ---
                prompt = (
                    f"Summarize this recall information for someone considering buying this used vehicle. "
                    f"Focus on major safety risks or repair requirements, and highlight what the buyer should pay most attention to.\n"
                    f"{recall_info}"
                )
                llm_response = await agent.arun(prompt)
                print("Agent summary:", llm_response)
            else:
                print("Could not parse vehicle info from your prompt. Please enter as '<year> <make> <model>'.")
        except Exception as e:
            print("Error:", e)
        except KeyboardInterrupt:
            print("Exiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())




