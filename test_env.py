from src.utils.config import validate_settings, settings

validate_settings()

print("Environment variables loaded successfully.")
print("Chat deployment:", settings.AZURE_OPENAI_CHAT_DEPLOYMENT)
print("Embedding deployment:", settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
print("Search index:", settings.SEARCH_INDEX_NAME)