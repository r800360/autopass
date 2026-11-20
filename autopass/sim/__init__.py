"""Lightweight kinematic simulator (pure standard library).

Stands in for CARLA where a GPU simulator is impractical: unit tests, CI, Monte-Carlo
evaluation over thousands of sampled scenarios, and the in-browser demo. It implements the
same World protocol as the CARLA backend, so the agent, critic and controller are identical.
"""
