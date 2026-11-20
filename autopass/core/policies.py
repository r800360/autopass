"""Policy configuration shared by the simulator, CARLA runner and browser."""
from dataclasses import replace

from .episode import Policy

POLICIES = ("autopass", "no_pass", "always_pass", "first_release")
SIM_POLICIES = (*POLICIES, "no_critic")


def named_policy(name: str, *, steering: str = "waypoint", realtime: bool = False) -> Policy:
    if name == "first_release":
        policy = Policy.first_release()
        policy.name = name
    elif name == "always_pass":
        policy = Policy(name=name, abort=False)
    elif name == "no_critic":
        policy = Policy(name=name)
        policy.agent.use_critic = False
    elif name in ("autopass", "no_pass", "llm_direct"):
        policy = Policy(name=name)
    else:
        raise ValueError(f"unknown policy: {name}")
    return replace(policy, steering=steering, agent=replace(policy.agent, realtime=realtime))
