"""Quantum-Inspired Simulated Annealing (QISA).

This module provides a reusable optimization engine that can be applied to:
- exam timetable generation
- weekly class timetable generation
- staff allocation / workload balancing

Design goals (college-project friendly):
- clear, explainable implementation
- works on a normal laptop (no quantum hardware)
- modular: the optimizer is problem-agnostic

Quantum-inspired aspect
----------------------
Classic SA accepts some worse moves with probability exp(-ΔE/T).
Quantum annealing intuition suggests that "tunneling" can help cross high but
*thin* barriers. In a classical heuristic, we mimic that by occasionally allowing
larger uphill moves than classical SA would at the same temperature.

We implement that with two acceptance channels:
1) thermal channel: exp(-ΔE/T)
2) tunneling channel (heavier tail): exp(-sqrt(ΔE)/(Γ*T))

Where Γ (gamma) is a user-controlled "tunneling strength".

The optimizer minimizes an energy function E(state).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar

import math
import random


TState = TypeVar("TState")


class NeighborFn(Protocol[TState]):
    def __call__(self, state: TState, rng: random.Random) -> TState:  # pragma: no cover
        """Return a randomly sampled neighbor of `state`."""


class EnergyFn(Protocol[TState]):
    def __call__(self, state: TState) -> float:  # pragma: no cover
        """Return energy/cost to MINIMIZE."""


class CallbackFn(Protocol[TState]):
    def __call__(
        self,
        step: int,
        temperature: float,
        current_state: TState,
        current_energy: float,
        best_state: TState,
        best_energy: float,
        accepted: bool,
    ) -> None:  # pragma: no cover
        """Optional progress callback called each step."""


@dataclass(frozen=True)
class AnnealConfig:
    """Configuration for QISA.

    Parameters are chosen to be understandable by students.

    Attributes:
        steps: Total number of iterations.
        t_start: Initial temperature.
        t_end: Final temperature (near zero, but not exactly zero).
        gamma: Tunneling strength. Higher => more likely to accept uphill moves.
        reheats: Number of restarts/reheats (0 disables). A reheat resets the
            temperature schedule but starts from the best-so-far state.
        seed: RNG seed for reproducibility.
    """

    steps: int = 50_000
    t_start: float = 5.0
    t_end: float = 0.05
    gamma: float = 1.5
    reheats: int = 1
    seed: Optional[int] = 42


@dataclass
class AnnealResult(Generic[TState]):
    best_state: TState
    best_energy: float
    best_step: int
    accepted_moves: int
    total_steps: int


def _geom_temperature(step: int, steps: int, t_start: float, t_end: float) -> float:
    """Geometric cooling schedule."""

    if steps <= 1:
        return t_end
    frac = step / (steps - 1)
    # T = T_start * (T_end/T_start)^frac
    return t_start * ((t_end / t_start) ** frac)


def _accept_prob(delta_e: float, temperature: float, gamma: float) -> float:
    """Quantum-inspired acceptance probability for an uphill move.

    delta_e is positive here (candidate_energy - current_energy).
    """

    if temperature <= 0:
        return 0.0

    # thermal (classic SA)
    p_thermal = math.exp(-delta_e / temperature)

    # tunneling-like heavier-tail acceptance
    # - uses sqrt(ΔE) to reduce sensitivity for large but "thin" barriers
    # - scaled by gamma which acts like a tunneling field strength
    # guard: gamma very small -> behave like classic SA
    g = max(gamma, 1e-9)
    p_tunnel = math.exp(-math.sqrt(delta_e) / (g * temperature))

    # Combine channels via union probability
    # P = 1 - (1-p1)(1-p2)
    return 1.0 - (1.0 - p_thermal) * (1.0 - p_tunnel)


def anneal(
    initial_state: TState,
    neighbor: NeighborFn[TState],
    energy: EnergyFn[TState],
    config: AnnealConfig = AnnealConfig(),
    callback: Optional[CallbackFn[TState]] = None,
) -> AnnealResult[TState]:
    """Run quantum-inspired simulated annealing.

    Contract:
    - Minimizes `energy(state)`
    - `neighbor` must return a valid state

    Returns:
        AnnealResult with best_state and metrics.
    """

    rng = random.Random(config.seed)

    def run_one_pass(start_state: TState, pass_idx: int) -> AnnealResult[TState]:
        steps = config.steps
        current = start_state
        current_e = energy(current)

        best = current
        best_e = current_e
        best_step = 0
        accepted_moves = 0

        for step in range(steps):
            t = _geom_temperature(step, steps, config.t_start, config.t_end)
            cand = neighbor(current, rng)
            cand_e = energy(cand)

            accepted = False
            if cand_e <= current_e:
                accepted = True
            else:
                delta = cand_e - current_e
                p = _accept_prob(delta, t, config.gamma)
                if rng.random() < p:
                    accepted = True

            if accepted:
                current = cand
                current_e = cand_e
                accepted_moves += 1

                if current_e < best_e:
                    best = current
                    best_e = current_e
                    best_step = step + pass_idx * steps

            if callback is not None:
                callback(
                    step=step + pass_idx * steps,
                    temperature=t,
                    current_state=current,
                    current_energy=current_e,
                    best_state=best,
                    best_energy=best_e,
                    accepted=accepted,
                )

        return AnnealResult(
            best_state=best,
            best_energy=best_e,
            best_step=best_step,
            accepted_moves=accepted_moves,
            total_steps=steps,
        )

    # First pass
    result = run_one_pass(initial_state, pass_idx=0)

    # Reheats/restarts from best
    best_state = result.best_state
    best_energy = result.best_energy
    best_step = result.best_step
    accepted_moves = result.accepted_moves
    total_steps = result.total_steps

    for i in range(config.reheats):
        rerun = run_one_pass(best_state, pass_idx=i + 1)
        accepted_moves += rerun.accepted_moves
        total_steps += rerun.total_steps
        if rerun.best_energy < best_energy:
            best_state = rerun.best_state
            best_energy = rerun.best_energy
            best_step = rerun.best_step

    return AnnealResult(
        best_state=best_state,
        best_energy=best_energy,
        best_step=best_step,
        accepted_moves=accepted_moves,
        total_steps=total_steps,
    )
