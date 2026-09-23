# test_openai.py

from openai import AzureOpenAI
from src.utils.config import settings

client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
)

response = client.chat.completions.create(
    model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
    messages=[
        {"role": "user", "content": "Say Hello!"}
    ]
)

print(response.choices[0].message.content)