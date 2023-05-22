"""Azure client."""
import logging
import os
from typing import Any, Dict, Optional, Type

from manifest.clients.openai import OPENAI_ENGINES, OpenAIClient
from manifest.request import LMRequest, Request

logger = logging.getLogger(__name__)

# Azure deployment name can only use letters and numbers, no spaces. Hyphens ("-") and
# underscores ("_") may be used, except as ending characters. We create this mapping to
# handle difference between Azure and OpenAI
AZURE_DEPLOYMENT_NAME_MAPPING = {
    "gpt-3.5-turbo": "gpt-35-turbo",
    "gpt-3.5-turbo-0301": "gpt-35-turbo-0301",
}
OPENAI_DEPLOYMENT_NAME_MAPPING = {
    "gpt-35-turbo": "gpt-3.5-turbo",
    "gpt-35-turbo-0301": "gpt-3.5-turbo-0301",
}


class AzureClient(OpenAIClient):
    """Azure client."""

    PARAMS = OpenAIClient.PARAMS
    REQUEST_CLS: Type[Request] = LMRequest
    NAME = "azureopenai"

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the AzureOpenAI server.

        connection_str is passed as default AZURE_OPENAI_KEY if variable not set.

        Args:
            connection_str: connection string.
            client_args: client arguments.
        """
        connection_parts = connection_str.split("::")
        if len(connection_parts) == 1:
            self.api_key = connection_parts[0]
        elif len(connection_parts) == 2:
            self.api_key, self.host = connection_parts
        else:
            raise ValueError(
                "Invalid connection string. "
                "Must be either AZURE_OPENAI_KEY or "
                "AZURE_OPENAI_KEY::AZURE_OPENAI_ENDPOINT"
            )
        self.api_key = self.api_key or os.environ.get("AZURE_OPENAI_KEY")
        if self.api_key is None:
            raise ValueError(
                "AzureOpenAI API key not set. Set AZURE_OPENAI_KEY environment "
                "variable or pass through `client_connection`."
            )
        self.host = self.host or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.host = self.host.rstrip("/")
        if self.host is None:
            raise ValueError(
                "Azure Service URL not set "
                "(e.g. https://openai-azure-service.openai.azure.com/)."
                " Set AZURE_OPENAI_ENDPOINT or pass through `client_connection`."
                " as AZURE_OPENAI_KEY::AZURE_OPENAI_ENDPOINT"
            )
        for key in self.PARAMS:
            setattr(self, key, client_args.pop(key, self.PARAMS[key][1]))
        if getattr(self, "engine") not in OPENAI_ENGINES:
            raise ValueError(
                f"Invalid engine {getattr(self, 'engine')}. Must be {OPENAI_ENGINES}."
            )

    def get_generation_url(self) -> str:
        """Get generation URL."""
        engine = getattr(self, "engine")
        deployment_name = AZURE_DEPLOYMENT_NAME_MAPPING.get(engine, engine)
        return (
            self.host
            + "/openai/deployments/"
            + deployment_name
            + "/completions?api-version=2023-05-15"
        )

    def get_generation_header(self) -> Dict[str, str]:
        """
        Get generation header.

        Returns:
            header.
        """
        return {"api-key": f"{self.api_key}"}

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        # IMPORTANT!!!
        # Azure models are the same as openai models. So we want to unify their
        # cached. Make sure we retrun the OpenAI name here.
        return {"model_name": OpenAIClient.NAME, "engine": getattr(self, "engine")}
