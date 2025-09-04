import os
import json
import requests
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv

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
    }
]

available_functions = {
    "get_name_for_day": get_name_for_day,
}

# Function to process messages and handle function calls
def get_completion_from_messages(messages, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,  # Custom tools
        tool_choice="auto"  # Allow AI to decide if a tool should be called
    )

    response_message = response.choices[0].message

    print("First response:", response_message)

    if response_message.tool_calls:
        # Find the tool call content
        tool_call = response_message.tool_calls[0]

        # Extract tool name and arguments
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id

        # Call the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        print(function_response)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args),
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        final_answer = second_response.choices[0].message

        print("Second response:", final_answer)
        return final_answer

    return response_message

# Example usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Kdo měl svátek 28. října 2024?"},
    #{"role": "user", "content": "Napiš ASDF"},
]

response = get_completion_from_messages(messages)
print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
content = getattr(response, "content", None)
if content:
    print(content)
else:
    print("No text content in the response")
