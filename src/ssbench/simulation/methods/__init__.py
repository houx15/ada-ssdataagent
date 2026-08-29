"""Generation methods. A method turns (dataset spec, conditioning inputs) into a synthetic DataFrame."""

from ssbench.simulation.methods.base import GenerationMethod, create_method, register_method
from ssbench.simulation.methods.direct import DirectGeneration

__all__ = ["GenerationMethod", "register_method", "create_method", "DirectGeneration"]
