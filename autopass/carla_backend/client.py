"""Connect to a user-started CARLA 0.9.16 server over standard RPC."""
from __future__ import annotations

import os


def connect(timeout_s: float = 30.0, *, host: str | None = None, port: int | None = None):
    import carla

    host = host or os.environ.get("CARLA_HOST", "127.0.0.1")
    port = port if port is not None else int(os.environ.get("CARLA_PORT", "2000"))
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    version = client.get_server_version()
    if version != "0.9.16" or client.get_client_version() != version:
        raise RuntimeError(f"CARLA 0.9.16 client/server required. Server reported {version}")
    return client


def current_town(client) -> str:
    return client.get_world().get_map().name.split("/")[-1].removesuffix("_Opt")


def load_town(client, town: str, *, reload: bool = False):
    """Load the requested map only with an explicit request from the caller."""
    world = client.get_world()
    if current_town(client) == town:
        return world
    if not reload:
        raise RuntimeError(f"Server has {world.get_map().name}, expected {town}. Pass --load-town on a dedicated server.")
    client.set_timeout(120.0)
    try:
        available = {name.split("/")[-1] for name in client.get_available_maps()}
        candidate = f"{town}_Opt" if f"{town}_Opt" in available else town
        if candidate not in available:
            raise RuntimeError(f"{town} is missing from this CARLA server")
        return client.load_world(candidate)
    finally:
        client.set_timeout(30.0)
