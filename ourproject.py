# copied/pasted directly from PerplexityAI

import os
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


print(os.environ.get("HF_TOKEN"))
print(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))


from transformers import pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_auth_token="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"
)





import asyncio
from fairlib import (
    HuggingFaceAdapter, ToolRegistry, SafeCalculatorTool, ToolExecutor,
    WorkingMemory, ReActPlanner, SimpleAgent, RoleDefinition
)

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
    agent = SimpleAgent(llm=llm, planner=planner, tool_executor=executor,
                        memory=working_memory, max_steps=10)
    return agent

async def main():
    # Specialized agent roles for car buying advice
    agent_prompts = [
        ("You are a vehicle research assistant. Provide comprehensive specs, reliability, safety ratings, and notable features for the car model requested by the user. Reason step by step and cite relevant consumer sources if possible."),
        ("You are a car market analyst. Offer advice on pricing, resale value, maintenance costs, insurance trends, and buying tips for the specified vehicle, tailored to North American car buyers.")
    ]
    # Optional: register tools to assist analysis (calculator for pricing, future tool for lookup, etc.)
    calculator_tool = SafeCalculatorTool()
    agents = [
        create_agent("TinyLlama/TinyLlama-1.1B-Chat-v1.0", agent_prompts[0], tools=[calculator_tool]),
        create_agent("TinyLlama/TinyLlama-1.1B-Chat-v1.0", agent_prompts[1], tools=[calculator_tool]),
    ]
    print("Car Buying Information Multi-agent System")
    print("Type the car (make, model, year, etc.) you want info about! (Type 'exit' to quit)")

    while True:
        user_input = input("Vehicle Query: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        # The first agent provides vehicle research
        research_info = await agents[0].arun(user_input)
        print(f"\n[Vehicle Research Agent] {research_info}")

        # The second agent gives market analysis based on the first agent's output and user context
        market_info = await agents[1].arun(research_info)
        print(f"[Market Analysis Agent] {market_info}\n")

if __name__ == "__main__":
    asyncio.run(main())
