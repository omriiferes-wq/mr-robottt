import os
import requests
import gradio as gr

API_KEY = os.getenv("NOTRACK_API_KEY")
API_URL = "https://api.notrack.ai/v1/chat/completions"
MODEL = "notrack-uncensored"

SYSTEM_PROMPT = """
You are MrROBOT, a helpful AI assistant.
Answer clearly, accurately and respectfully.
Remember the conversation context.
"""

def predict(message, history):
    if not API_KEY:
        return "Error: API key is missing. Check your hosting Secrets."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    history = history or []

    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            if isinstance(user_message, str):
                messages.append({"role": "user", "content": user_message})
            if isinstance(assistant_message, str):
                messages.append({"role": "assistant", "content": assistant_message})
        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in ["user", "assistant"] and isinstance(content, str):
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "Error: API connection timed out."
    except requests.exceptions.HTTPError as error:
        return "API HTTP Error: " + str(error)
    except requests.exceptions.RequestException as error:
        return "Connection Error: " + str(error)
    except (KeyError, IndexError, ValueError):
        return "Error: Invalid API response."
    except Exception as error:
        return "Unexpected Error: " + str(error)


demo = gr.ChatInterface(fn=predict)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
