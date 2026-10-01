import logging
import httpx
from typing import Optional, Dict, Any

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Client for interacting with open-source LLMs hosted via Ollama or compatible APIs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.DEFAULT_LLM_MODEL
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return list of models available in the local Ollama instance."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
        return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Synchronously generate text response from the LLM.
        A low temperature (0.1) is used by default to enforce factual grounding.
        """
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"Ollama API returned error {response.status_code}: {error_detail}")
                    raise RuntimeError(f"Ollama error ({response.status_code}): {error_detail}")

                result = response.json()
                return result.get("response", "").strip()

        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Please ensure 'ollama serve' is running."
            )
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            raise
