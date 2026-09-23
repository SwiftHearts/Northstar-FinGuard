import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
    SEARCH_KEY = os.getenv("SEARCH_KEY")
    SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME", "northstar-finguard-index")

    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "northstar-finguard")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")

# Hold the configuration settings in a single instance for easy access throughout the application
# e.g settings.AZURE_OPENAI_API_KEY, settings.SEARCH_ENDPOINT, etc.
settings = Settings()


def validate_settings():
    required = {
        "AZURE_OPENAI_ENDPOINT": settings.AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": settings.AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_CHAT_DEPLOYMENT": settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "SEARCH_ENDPOINT": settings.SEARCH_ENDPOINT,
        "SEARCH_KEY": settings.SEARCH_KEY,
        "SEARCH_INDEX_NAME": settings.SEARCH_INDEX_NAME,
    }

    missing = [key for key, value in required.items() if not value]

    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

    return True