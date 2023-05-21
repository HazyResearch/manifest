"""Client connection."""
import logging
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Extra

from manifest.clients.ai21 import AI21Client
from manifest.clients.azureopenai import AzureClient
from manifest.clients.azureopenai_chat import AzureChatClient
from manifest.clients.client import Client
from manifest.clients.cohere import CohereClient
from manifest.clients.dummy import DummyClient
from manifest.clients.google import GoogleClient
from manifest.clients.google_chat import GoogleChatClient
from manifest.clients.huggingface import HuggingFaceClient
from manifest.clients.huggingface_embedding import HuggingFaceEmbeddingClient
from manifest.clients.openai import OpenAIClient
from manifest.clients.openai_chat import OpenAIChatClient
from manifest.clients.openai_embedding import OpenAIEmbeddingClient
from manifest.clients.toma import TOMAClient
from manifest.connections.scheduler import RandomScheduler, RoundRobinScheduler

logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CLIENT_CONSTRUCTORS = {
    AI21Client.NAME: AI21Client,
    AzureClient.NAME: AzureClient,
    AzureChatClient.NAME: AzureChatClient,
    CohereClient.NAME: CohereClient,
    DummyClient.NAME: DummyClient,
    GoogleClient.NAME: GoogleClient,
    GoogleChatClient.NAME: GoogleChatClient,
    HuggingFaceClient.NAME: HuggingFaceClient,
    HuggingFaceEmbeddingClient.NAME: HuggingFaceEmbeddingClient,
    OpenAIClient.NAME: OpenAIClient,
    OpenAIChatClient.NAME: OpenAIChatClient,
    OpenAIEmbeddingClient.NAME: OpenAIEmbeddingClient,
    TOMAClient.NAME: TOMAClient,
}

CLIENT_REQUEST_TYPES: Dict[str, Type] = {
    k: v.REQUEST_CLS for k, v in CLIENT_CONSTRUCTORS.items()
}

# Diffusion
DIFFUSION_CLIENTS = ["diffuser", "tomadiffuser"]
try:
    from manifest.clients.diffuser import DiffuserClient
    from manifest.clients.toma_diffuser import TOMADiffuserClient

    CLIENT_CONSTRUCTORS[DiffuserClient.NAME] = DiffuserClient
    CLIENT_CONSTRUCTORS[TOMADiffuserClient.NAME] = TOMADiffuserClient
except Exception:
    logger.info("Diffusion not supported. Skipping import.")
    pass

SCHEDULER_CONSTRUCTORS = {
    RandomScheduler.NAME: RandomScheduler,
    RoundRobinScheduler.NAME: RoundRobinScheduler,
}


class Timing(BaseModel):
    """Timing class."""

    start: float = -1.0
    end: float = -1.0


class ClientConnection(BaseModel):
    """Client Connection class."""

    client_name: str
    # Use environment variables (depending on client)
    client_connection: Optional[str] = None
    # Use default engine
    engine: Optional[str] = None

    # Prevent extra args
    class Config:
        """Config class.

        Allows to override pydantic behavior.
        """

        extra = Extra.forbid


class ClientConnectionPool:
    """Client connection pool."""

    def __init__(
        self,
        client_pool: List[ClientConnection],
        client_pool_scheduler: str = "round_robin",
        client_args: Dict[str, Any] = {},
    ):
        """Init."""
        # Verify the clients are allowed and supported
        for client in client_pool:
            if client.client_name not in CLIENT_CONSTRUCTORS:
                if client.client_name in DIFFUSION_CLIENTS:
                    raise ImportError(
                        f"Diffusion client {client.client_name} requires "
                        "the proper install. Make sure to run "
                        "`pip install manifest-ml[diffusers]` "
                        "or install Pillow."
                    )
                else:
                    raise ValueError(
                        f"Unknown client name: {client.client_name}. "
                        f"Choices are {list(CLIENT_CONSTRUCTORS.keys())}"
                    )
        # Verify that the serialization of all clients is the same
        request_types = set(
            [CLIENT_REQUEST_TYPES[client.client_name] for client in client_pool]
        )
        if len(request_types) > 1:
            raise ValueError(
                "All clients in the client pool must use the same request type. "
                f"You have {sorted(list(map(str, request_types)))}"
            )

        # Verify scheduler
        if client_pool_scheduler not in SCHEDULER_CONSTRUCTORS:
            raise ValueError(f"Unknown scheduler: {client_pool_scheduler}.")

        self.request_type = request_types.pop()
        # Initialize the clients
        # We must keep track of the used args so we know
        # if a user passed in an arg that was never used
        used_args = set()
        self.client_pool = []
        for client in client_pool:
            to_pass_kwargs = client_args.copy()
            # Override the engine param for each
            to_pass_kwargs.pop("engine", None)
            if client.engine:
                to_pass_kwargs["engine"] = client.engine
            self.client_pool.append(
                CLIENT_CONSTRUCTORS[client.client_name](  # type: ignore
                    client.client_connection, client_args=to_pass_kwargs
                )
            )
            # Udpate used args
            for k in client_args:
                if k not in to_pass_kwargs:
                    used_args.add(k)
        # Removed used args
        for k in used_args:
            client_args.pop(k)

        # Get the scheduler
        self.scheduler = SCHEDULER_CONSTRUCTORS[client_pool_scheduler](
            num_clients=len(self.client_pool)
        )
        self.current_client_id = 0
        # Record timing metrics for each client for load balancing
        # TODO: Implement this in the future
        self.client_pool_metrics = [Timing() for _ in self.client_pool]

    def close(self) -> None:
        """Close."""
        for client in self.client_pool:
            client.close()

    def num_clients(self) -> int:
        """Get number of clients."""
        return len(self.client_pool)

    def get_next_client(self) -> Client:
        """Get client."""
        client_int = self.scheduler.get_client()
        self.current_client_id = client_int
        return self.client_pool[client_int]

    def get_current_client(self) -> Client:
        """Get current client."""
        return self.client_pool[self.current_client_id]

    def start_timer(self) -> None:
        """Start timer."""
        self.client_pool_metrics[self.current_client_id].start = time.time()

    def end_timer(self) -> None:
        """End timer."""
        self.client_pool_metrics[self.current_client_id].end = time.time()
