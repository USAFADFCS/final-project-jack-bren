import os
import asyncio
from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent,
    RoleDefinition
)

# Set your Hugging Face token, replace 'your_token' with your actual token
os.environ["HF_TOKEN"] = "your_token"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token"

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
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=working_memory,
        max_steps=10
    )
    return agent

async def main():
    print("Initializing a vehicle recall/issues agent for demonstration...")
    # Agent role prompt focused on vehicle issues and recalls
    role_prompt = (
        "You are an expert auto safety and reliability assistant. "
        "Identify and summarize known potential problems, manufacturer recalls, safety notices, and common issues for the vehicle the user enters. "
        "Use detailed reasoning and cite trusted car safety or recall databases if possible."
    )

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # No tools required for recall/issues identification, but tools can be added if needed
    agent = create_agent(model_name, role_prompt)

    print("Agent created. Enter a vehicle make, model, and year (e.g. '2016 Honda Civic'). Type exit to quit.")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Agent: Goodbye!")
                break
            response = await agent.arun(user_input)
            print(f"Agent: {response}")
        except KeyboardInterrupt:
            print("Agent: Exiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())

