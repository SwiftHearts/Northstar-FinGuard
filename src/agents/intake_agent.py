"""Intake agent: normalizes a submitted marketing/client-communication draft and classifies
its channel and audience so downstream compliance rules can be applied correctly.

Usage (standalone test):
    python -m src.agents.intake_agent
"""

# JSON used for converting the JSON response from the OpenAI API into a Python dictionary. 
# The response is expected to be a JSON object with "channel" and "audience" keys, which are then validated against predefined valid values.
import json

# UUID is used to generate a unique identifier for each draft communication, which is included 
# in the metadata returned by the intake agent.
import uuid

# The typing module's get_args function is used to retrieve the valid values for the Channel and Audience
# enums defined in the src.graph.state module. This allows the intake agent to validate the classification
# results against the expected values.
from typing import get_args

# The src.graph.state module defines the Audience, Channel, ComplianceState, and IntakeMetadata types 
# used in the intake agent.
from src.graph.state import Audience, Channel, ComplianceState, IntakeMetadata

# The get_azure_openai_client function is imported from the src.utils.azure_clients module.
from src.utils.azure_clients import get_azure_openai_client

# Access configuration values such as API keys, endpoints, and model deployment names.
from src.utils.config import settings

# Get the valid values for the Channel and Audience enums defined in the src.graph.state module.
VALID_CHANNELS = get_args(Channel)
VALID_AUDIENCES = get_args(Audience)

CLASSIFICATION_PROMPT = """You are a compliance intake assistant for a registered investment adviser.
Classify the following draft communication.

Most drafts carry some stylistic signal of their intended channel even without an explicit \
label -- use it rather than defaulting to "other":
- Salutations like "Dear [name]" or a signature block suggest email.
- Hashtags, emoji, short punchy sentences, or explicit calls to "follow" or "like" suggest \
social_media.
- Formal, printed-copy phrasing with no direct address (third-person, brochure-style \
sentences) suggests brochure.
- References to slides, talking points, or being presented to a specific prospect/investor \
suggests pitch_deck.
- Phrasing inviting someone to "join," "register," or "attend" a live session suggests webinar.
- Only use "other" when the text truly has none of these cues -- do not use it just because \
no channel is explicitly stated.

Return strict JSON with exactly these keys:
- "channel": one of {channels}
- "audience": one of {audiences}

Draft:
---
{text}
---
"""

# Helper function for internal use by the intake_agent function. It sends the draft text to the Azure OpenAI 
# API for classification, and returns a dictionary containing the classified channel and audience. 
# If the classification results are not valid, default to "other" for channel and "mixed" for audience.
def _classify(text: str) -> dict:
    # Create an Azure OpenAI client using the factory function defined in src.utils.azure_clients.
    client = get_azure_openai_client()
    # Chats completion request to the Azure OpenAI API using the specified model deployment and prompt. 
    # The response is expected to be a JSON object containing the classified channel and audience.
    response = client.chat.completions.create(
        # The model deployment name is retrieved from the settings module, which is configured in src.utils.config.
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        # The messages parameter contains the prompt for the classification task, formatted with the draft text 
        # and valid channel/audience values.
        messages=[
            {
                # Model receives the user message
                "role": "user",
                # The content of the message is the classification prompt, formatted with the draft text and 
                # valid channel/audience values.
                "content": CLASSIFICATION_PROMPT.format(
                    text=text, channels=list(VALID_CHANNELS), audiences=list(VALID_AUDIENCES)
                ),
            }
        ],
        # The response_format parameter specifies that the response should be returned as a JSON object,
        # and the temperature parameter controls the randomness of the model's output (0 means deterministic).
        response_format={"type": "json_object"},
        temperature=0,
    )
    # Turn the response from a string into a JSON object containing the classified 
    # channel and audience.
    result = json.loads(response.choices[0].message.content)

    # Ensure that the classified channel and audience are valid values defined in the Channel and Audience enums.
    channel = result.get("channel") if result.get("channel") in VALID_CHANNELS else "other"
    audience = result.get("audience") if result.get("audience") in VALID_AUDIENCES else "mixed"
    return {"channel": channel, "audience": audience}

# Define the LangGraph node (the intake_agent) function that takes a ComplianceState object as input and 
# returns a dictionary containing the normalized source text and metadata.
def intake_agent(state: ComplianceState) -> dict:
    # Normalize the source text by stripping leading/trailing whitespace and validating that it is not empty.
    text = state["source_text"].strip()
    # If the source text is empty after stripping whitespace, raise a ValueError to indicate that there is 
    # nothing to review.
    if not text:
        raise ValueError("source_text is empty; nothing to review")

    # Classify the normalized source text using the _classify helper function, which sends the text to the
    # Azure OpenAI API for classification and returns the classified channel and audience.
    classification = _classify(text)

    # Create a metadata dictionary containing a unique draft ID, the classified channel and audience,
    # and the word count of the normalized source text. The draft ID is generated using the uuid 
    # module to ensure uniqueness.
    metadata: IntakeMetadata = {
        "draft_id": uuid.uuid4().hex[:12],
        "channel": classification["channel"],
        "audience": classification["audience"],
        "word_count": len(text.split()),
    }
    # Add to the LangGraph state. Return a dictionary containing the normalized source text and the 
    # metadata dictionary, which can be used by downstream compliance rules to apply the appropriate 
    # checks and validations based on the classified channel and audience.
    return {"source_text": text, "metadata": metadata}

# Standalone test: run the intake_agent function with a sample draft communication and print the result.
if __name__ == "__main__":
    sample = (
        "Northstar Wealth Partners is proud to say our growth portfolio beat the market "
        "every year for the last decade. Sign up today for free portfolio management!"
    )
    result = intake_agent({"source_text": sample})
    print(json.dumps(result, indent=2))
