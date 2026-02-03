"""
Phillip - Neural network Melee AI fighter adapter.

This adapter integrates vladfi1's Phillip (slippi-ai) neural network AI
into the nojohns fighter system.

Phillip is trained via imitation learning on human Slippi replays, then
refined through self-play reinforcement learning. It plays in a human-like
style with consistent reaction delays.
"""

from .phillip_fighter import PhillipFighter, PhillipConfig, load_phillip

__all__ = ['PhillipFighter', 'PhillipConfig', 'load_phillip']
