"""PINN-Options implementation package.

The NumPy baselines remain importable when the optional PyTorch dependency is
not installed (useful for lightweight evaluation and documentation builds).
"""

from .baselines import black_scholes_call, black_scholes_delta, black_scholes_gamma

try:  # Keep CPU/NumPy baselines usable without installing the training stack.
    from .model import MLP, PINN
except ImportError:  # pragma: no cover - depends on the environment
    MLP = PINN = None

__all__ = ["MLP", "PINN", "black_scholes_call", "black_scholes_delta", "black_scholes_gamma"]
