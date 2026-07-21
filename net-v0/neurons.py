"""Base neuron models for the LiSNN net-v0 implementation.

The neuron classes expose ``neuron_type`` as structural metadata for the
network/weight-matrix layer:

- ``"e"``: excitatory
- ``"I"``: inhibitory

The neuron dynamics deliberately do not alter their equations based on that
metadata. Synaptic sign and E/I connectivity constraints belong to the weight
matrix, not the membrane model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite
from typing import Literal, TypeAlias


NeuronType: TypeAlias = Literal["e", "I"]
_VALID_NEURON_TYPES = frozenset(("e", "I"))
_MAX_EXP_ARGUMENT = 50.0


def _validate_neuron_type(neuron_type: str) -> NeuronType:
    """Validate and preserve the explicit E/I neuron marker."""

    if neuron_type not in _VALID_NEURON_TYPES:
        raise ValueError(
            "neuron_type must be exactly 'e' (excitatory) or 'I' (inhibitory)"
        )
    return neuron_type  # type: ignore[return-value]


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return value


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(slots=True)
class LIF:
    """Single leaky integrate-and-fire neuron.

    ``step()`` preserves the two-value convention used by the recovered
    time-code/frame baseline:

    ``(pre_reset_potential, stored_membrane_potential)``

    ``pre_reset_potential`` is the membrane value before a threshold-triggered
    reset. ``potential`` stores either that value or ``v_reset`` after firing.
    The Boolean firing state is available through ``last_spike``.

    ``neuron_type`` is metadata only. It is intended for the future weight
    matrix to determine whether outgoing synapses are excitatory or inhibitory.
    """

    neuron_type: NeuronType = "e"
    dt: float = 1.0
    tau_m: float = 20.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_threshold: float = -55.0
    potential: float | None = None
    last_spike: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.neuron_type = _validate_neuron_type(self.neuron_type)
        self.dt = _require_positive("dt", self.dt)
        self.tau_m = _require_positive("tau_m", self.tau_m)
        self.v_rest = _require_finite("v_rest", self.v_rest)
        self.v_reset = _require_finite("v_reset", self.v_reset)
        self.v_threshold = _require_finite("v_threshold", self.v_threshold)

        if self.potential is None:
            self.potential = self.v_rest
        else:
            self.potential = _require_finite("potential", self.potential)

    @property
    def is_excitatory(self) -> bool:
        return self.neuron_type == "e"

    @property
    def is_inhibitory(self) -> bool:
        return self.neuron_type == "I"

    def reset(self, potential: float | None = None) -> None:
        """Reset membrane state without changing neuron parameters or type."""

        self.potential = (
            self.v_rest
            if potential is None
            else _require_finite("potential", potential)
        )
        self.last_spike = False

    def step(self, input_current: float) -> tuple[float, float]:
        """Advance the membrane by one integration step.

        Args:
            input_current:
                External plus recurrent input for this neuron.

        Returns:
            A pair ``(pre_reset_potential, stored_membrane_potential)``.
        """

        current = _require_finite("input_current", input_current)
        potential = float(self.potential)

        dv = (
            -(potential - self.v_rest) + current
        ) * (self.dt / self.tau_m)
        pre_reset_potential = potential + dv

        self.last_spike = pre_reset_potential >= self.v_threshold
        self.potential = (
            self.v_reset if self.last_spike else pre_reset_potential
        )

        return float(pre_reset_potential), float(self.potential)

    __call__ = step


@dataclass(slots=True)
class AdExLIF:
    """Single adaptive exponential integrate-and-fire neuron.

    The membrane follows the adaptive exponential integrate-and-fire equations:

    ``C * dV/dt = -gL(V - EL) + gL*DeltaT*exp((V - VT)/DeltaT) - w + I``

    ``tau_w * dw/dt = a(V - EL) - w``

    On a spike, the membrane resets to ``v_reset`` and adaptation increases by
    ``b``. As with :class:`LIF`, ``neuron_type`` is metadata for the weight
    matrix and does not change the membrane equation.

    ``step()`` returns the same two-value convention as :class:`LIF`:
    ``(pre_reset_potential, stored_membrane_potential)``.
    """

    neuron_type: NeuronType = "e"
    dt: float = 1.0

    capacitance: float = 200.0
    g_leak: float = 10.0
    v_rest: float = -65.0
    v_reset: float = -58.0
    v_threshold: float = -50.0
    delta_t: float = 2.0
    spike_cutoff: float = 20.0

    tau_w: float = 100.0
    a: float = 2.0
    b: float = 40.0

    potential: float | None = None
    adaptation: float = 0.0
    last_spike: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.neuron_type = _validate_neuron_type(self.neuron_type)
        self.dt = _require_positive("dt", self.dt)
        self.capacitance = _require_positive("capacitance", self.capacitance)
        self.g_leak = _require_positive("g_leak", self.g_leak)
        self.delta_t = _require_positive("delta_t", self.delta_t)
        self.tau_w = _require_positive("tau_w", self.tau_w)

        self.v_rest = _require_finite("v_rest", self.v_rest)
        self.v_reset = _require_finite("v_reset", self.v_reset)
        self.v_threshold = _require_finite("v_threshold", self.v_threshold)
        self.spike_cutoff = _require_finite("spike_cutoff", self.spike_cutoff)
        self.a = _require_finite("a", self.a)
        self.b = _require_finite("b", self.b)
        self.adaptation = _require_finite("adaptation", self.adaptation)

        if self.spike_cutoff <= self.v_threshold:
            raise ValueError("spike_cutoff must be greater than v_threshold")

        if self.potential is None:
            self.potential = self.v_rest
        else:
            self.potential = _require_finite("potential", self.potential)

    @property
    def is_excitatory(self) -> bool:
        return self.neuron_type == "e"

    @property
    def is_inhibitory(self) -> bool:
        return self.neuron_type == "I"

    def reset(
        self,
        potential: float | None = None,
        *,
        adaptation: float = 0.0,
    ) -> None:
        """Reset membrane and adaptation state without changing parameters."""

        self.potential = (
            self.v_rest
            if potential is None
            else _require_finite("potential", potential)
        )
        self.adaptation = _require_finite("adaptation", adaptation)
        self.last_spike = False

    def step(self, input_current: float) -> tuple[float, float]:
        """Advance the AdEx membrane and adaptation state by one step."""

        current = _require_finite("input_current", input_current)
        potential = float(self.potential)

        exponent_argument = min(
            _MAX_EXP_ARGUMENT,
            (potential - self.v_threshold) / self.delta_t,
        )
        exponential_current = (
            self.g_leak * self.delta_t * exp(exponent_argument)
        )

        dv_dt = (
            -self.g_leak * (potential - self.v_rest)
            + exponential_current
            - self.adaptation
            + current
        ) / self.capacitance
        dw_dt = (
            self.a * (potential - self.v_rest) - self.adaptation
        ) / self.tau_w

        pre_reset_potential = potential + self.dt * dv_dt
        next_adaptation = self.adaptation + self.dt * dw_dt

        self.last_spike = pre_reset_potential >= self.spike_cutoff
        if self.last_spike:
            self.potential = self.v_reset
            self.adaptation = next_adaptation + self.b
        else:
            self.potential = pre_reset_potential
            self.adaptation = next_adaptation

        if not isfinite(self.potential) or not isfinite(self.adaptation):
            raise FloatingPointError(
                "AdExLIF produced non-finite membrane or adaptation state"
            )

        return float(pre_reset_potential), float(self.potential)

    __call__ = step


__all__ = ["NeuronType", "LIF", "AdExLIF"]
