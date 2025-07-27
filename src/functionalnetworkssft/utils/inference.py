# %%
import os
from pathlib import Path
from openai import OpenAI

# %%
# Load .env file if it exists
env_file = Path("../../../.env")

if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Debug: Check if token is loaded and show first/last few characters
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"Token loaded: {hf_token[:8]}...{hf_token[-8:]}")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct",
    messages=[{"role": "user", "content": ""}],
)

print(completion.choices[0].message)
# %%
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],  # export HF_TOKEN=hf_...
)

response = client.chat.completions.create(
    model="microsoft/DialoGPT-medium",  # or any model from /v1/models
    messages=[{"role": "user", "content": "Hello from Hugging Face!"}],
    max_tokens=64,
)

print(response.choices[0].message.content)

# %%
from huggingface_hub import InferenceClient
import os

hf_token = os.environ["HF_TOKEN"]
client = InferenceClient(model="moonshotai/Kimi-K2-Instruct", token=hf_token)

response = client.chat_completion(
    messages=[{"role": "user", "content": "Tell me a joke"}], max_tokens=100
)

print(response.choices[0].message["content"])

# %%
