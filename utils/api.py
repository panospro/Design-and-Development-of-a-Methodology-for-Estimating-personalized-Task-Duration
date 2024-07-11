import os
from dotenv import load_dotenv
from groq import Groq
import json
import tiktoken

load_dotenv('.env')
api_key = os.environ.get("GROQ_API_KEY")

# Count the number of tokens for a given text
def count_tokens(text, tokenizer_name = 'cl100k_base'):
    encoding = tiktoken.get_encoding(tokenizer_name)
    tokens = encoding.encode(text)
    return len(tokens)

# Call groq and return the output as an object, i could also use llama3-8b-8192
def groq_call(message, model_name='llama3-70b-8192', max_response_tokens=2048, model_context_window = 8192):
    client = Groq(api_key=api_key)
    
    # Tokenize the input message to ensure it fits within the allowed limit
    total_input_tokens = sum(count_tokens(item['content']) for item in message)
    max_input_tokens = model_context_window - max_response_tokens
    if total_input_tokens > max_input_tokens:
        raise ValueError(f"Input message is too long. Maximum allowed tokens: {max_input_tokens}. The input tokens were: {total_input_tokens}")

    chat_completion = client.chat.completions.create(
        messages=message,
        model=model_name,
        temperature=0,
        max_tokens=max_response_tokens
    )

    return chat_completion.choices[0].message.content

def load_or_request_data(queries, filename, should_write_file=False):
    results = []
    loading_message_shown = False
    try:
        for i, query in enumerate(queries):
            new_filename = f"{filename}_{i + 1}.json"

            # Check if the file already exists
            if not should_write_file and os.path.exists(new_filename):
                if not loading_message_shown:
                    print(f"Loading data from {new_filename}")
                    loading_message_shown = True
                with open(new_filename, 'r') as file:
                    data = json.load(file)
            else:
                print(f"Making API call for query {i + 1}...")
                output = groq_call(query)
                print(output)
                data = json.loads(output)
                with open(new_filename, 'w') as file:
                    json.dump(data, file)

            results.append(data)

    except Exception as e:
        print(f"In api.py: {e}")
        return None

    return results
