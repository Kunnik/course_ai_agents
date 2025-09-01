import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Tool functions
def get_name_for_day(date: str):
    response = requests.get(f"https://svatkyapi.cz/api/day/{date}")
    if response.status_code == 200:
        data = response.json()
        name = data.get("name")
        return {"date": date, "name": name}
    else:
        return {"date": date, "name": "Error fetching data"}

def get_all_info_about_day(date: str):
    response = requests.get(f"https://svatkyapi.cz/api/day/{date}")
    if response.status_code == 200:
        data = response.json()
        return {"date": date, "data": data}
    else:
        return {"date": date, "data": "Error fetching data"}

def get_names_for_week(date: str):
    response = requests.get(f"https://svatkyapi.cz/api/week/{date}")
    if response.status_code == 200:
        data = response.json()
        weekData = data.map(lambda day: {
            day.get("name"),
            day.get("date"),
            day.get("dayInWeek")
        })
        return {"date": date, "weekData": weekData}
    else:
        return {"date": date, "weekData": "Error fetching data"}


# Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_name_for_day",
            "description": "Use this function to get the name associated with given day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    }
                },
                "required": ["date"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_info_about_day",
            "description": "Use this function to get all data about given day. Here are all fields: date, dayNumber, dayInWeek, monthNumber, month.nominative, month.genitive, year, name, isHoliday and holidayName.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    }
                },
                "required": ["date"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_names_for_week",
            "description": "Returns information about next week starting from given date. Fields for each day: date, dayInWeek, name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    }
                },
                "required": ["date"],
            },
        }
    }
]

available_functions = {
    "get_name_for_day": get_name_for_day,
    "get_all_info_about_day": get_all_info_about_day,
    "get_names_for_week": get_names_for_week,
}


class ReactAgent:
    """A ReAct (Reason and Act) agent that handles multiple tool calls."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.max_iterations = 10  # Prevent infinite loops

    def run(self, messages: List[Dict[str, Any]]) -> str:
        """
        Run the ReAct loop until we get a final answer.

        The agent will:
        1. Call the LLM
        2. If tool calls are returned, execute them
        3. Add results to conversation and repeat
        4. Continue until LLM returns only text (no tool calls)
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Call the LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
            )

            response_message = response.choices[0].message
            print(f"LLM Response: {response_message}")

            # Check if there are tool calls
            if response_message.tool_calls:
                # Add the assistant's message with tool calls to history
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response_message.tool_calls
                        ],
                    }
                )

                # Process ALL tool calls (not just the first one)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id

                    print(f"Executing tool: {function_name}({function_args})")

                    # Call the function
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)

                    print(f"Tool result: {function_response}")

                    # Add tool response to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": function_name,
                            "content": json.dumps(function_response),
                        }
                    )

                # Continue the loop to get the next response
                continue

            else:
                # No tool calls - we have our final answer
                final_content = response_message.content

                # Add the final assistant message to history
                messages.append({"role": "assistant", "content": final_content})

                print(f"\nFinal answer: {final_content}")
                return final_content

        # If we hit max iterations, return an error
        return "Error: Maximum iterations reached without getting a final answer."


def main():
    # Create a ReAct agent
    agent = ReactAgent()

    # Call 1 - This should call the simpler tool only
    print("=== Call 1: Single simple tool call ===")
    messages1 = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Kdo měl svátek 28. října 2024?"},
    ]

    result1 = agent.run(messages1.copy())
    print(f"\nResult: {result1}")

    # Call 2
    print("\n\n=== Call 2: Multiple call of same tool ===")
    messages2 = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Zjisti mi detalní informace o dnech 3. ledna 2024 a 3. ledna 2012?"},
    ]

    result2 = agent.run(messages2.copy())
    print(f"\nResult: {result2}")

    # Call 3
    print("\n\n=== Call 3: Test how smart is LLM when selecting tool ===")
    # LLM can call the week tool instead of calling the day tool twice
    messages3 = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Kdo měl svátek 28. a 30. října 2024?"},
    ]

    result3 = agent.run(messages3.copy())
    print(f"\nResult: {result3}")


if __name__ == "__main__":
    main()
