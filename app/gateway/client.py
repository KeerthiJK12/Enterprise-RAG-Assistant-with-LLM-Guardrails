import logfire

from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


PORTKEY_CONFIG_ID = "pc-enterp-e65d7d"


# Native Portkey client using the saved configuration.
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG_ID
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI.

    The saved Portkey Config controls:
    - Groq provider routing
    - fallback model
    - retry behaviour
    - caching

    ChatOpenAI is used because Portkey exposes an
    OpenAI-compatible API endpoint.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,

        # The saved Portkey Config handles the actual
        # model routing. Do not use @rag/... here.
        model="gpt-oss-120b",

        temperature=0,

        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_CONFIG_ID,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native
    client response headers.

    Returns MISS if the header cannot be found.
    """
    for attr in (
        "_raw_response",
        "_response",
        "_http_response"
    ):
        raw = getattr(response, attr, None)

        if raw is not None:
            status = getattr(
                raw,
                "headers",
                {}
            ).get(
                "x-portkey-cache-status",
                ""
            )

            if status:
                return status.upper()

    return "MISS"