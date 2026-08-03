"""Regret and time-uniform telemetry for online controllers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RegretSnapshot:
    rounds: int
    cumulative_loss: float
    best_fixed_comparator_loss: float
    static_regret: float
    dynamic_comparator_loss: float
    dynamic_regret: float


class RegretTracker:
    """Exact diagnostics for a finite comparator family."""

    def __init__(self, num_comparators: int) -> None:
        if num_comparators <= 0:
            raise ValueError("num_comparators must be positive")
        self._num_comparators = num_comparators
        self._learner_losses: list[float] = []
        self._comparator_losses: list[tuple[float, ...]] = []

    @property
    def rounds(self) -> int:
        return len(self._learner_losses)

    def update(self, learner_loss: float, comparator_losses: Sequence[float]) -> None:
        if len(comparator_losses) != self._num_comparators:
            raise ValueError("comparator_losses has the wrong length")
        values = tuple(float(value) for value in comparator_losses)
        if not math.isfinite(float(learner_loss)) or not all(math.isfinite(v) for v in values):
            raise ValueError("losses must be finite")
        self._learner_losses.append(float(learner_loss))
        self._comparator_losses.append(values)

    def snapshot(self) -> RegretSnapshot:
        if not self._learner_losses:
            return RegretSnapshot(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cumulative_loss = sum(self._learner_losses)
        totals = [
            sum(row[index] for row in self._comparator_losses)
            for index in range(self._num_comparators)
        ]
        best_fixed = min(totals)
        dynamic = sum(min(row) for row in self._comparator_losses)
        return RegretSnapshot(
            rounds=self.rounds,
            cumulative_loss=cumulative_loss,
            best_fixed_comparator_loss=best_fixed,
            static_regret=cumulative_loss - best_fixed,
            dynamic_comparator_loss=dynamic,
            dynamic_regret=cumulative_loss - dynamic,
        )

    def best_k_switch_comparator_loss(self, max_switches: int) -> float:
        """Return the exact best expert-sequence loss with at most K switches."""

        if max_switches < 0:
            raise ValueError("max_switches must be non-negative")
        if not self._comparator_losses:
            return 0.0

        inf = float("inf")
        first = self._comparator_losses[0]
        dp = [[inf] * self._num_comparators for _ in range(max_switches + 1)]
        for switches in range(max_switches + 1):
            dp[switches] = list(first)

        for row in self._comparator_losses[1:]:
            next_dp = [[inf] * self._num_comparators for _ in range(max_switches + 1)]
            for switches in range(max_switches + 1):
                for current in range(self._num_comparators):
                    stay = dp[switches][current]
                    change = inf
                    if switches > 0:
                        change = min(
                            dp[switches - 1][previous]
                            for previous in range(self._num_comparators)
                            if previous != current
                        )
                    next_dp[switches][current] = row[current] + min(stay, change)
            dp = next_dp
        return min(dp[max_switches])

    def tracking_regret(self, max_switches: int) -> float:
        return sum(self._learner_losses) - self.best_k_switch_comparator_loss(max_switches)

    def maximum_interval_static_regret(self) -> float:
        """Return the exact maximum static regret over all contiguous intervals."""

        rounds = self.rounds
        if rounds == 0:
            return 0.0
        best = -float("inf")
        for start in range(rounds):
            learner = 0.0
            comparator = [0.0] * self._num_comparators
            for end in range(start, rounds):
                learner += self._learner_losses[end]
                row = self._comparator_losses[end]
                for index in range(self._num_comparators):
                    comparator[index] += row[index]
                best = max(best, learner - min(comparator))
        return best


@dataclass
class AnytimeHoeffdingCS:
    """Conservative anytime-valid confidence sequence for a bounded mean.

    The construction allocates failure probability proportional to 1/t^2 and
    applies Hoeffding-Azuma plus a union bound. Validity requires bounded
    observations and the usual independent or martingale-difference mean model.
    """

    lower_bound: float
    upper_bound: float
    delta: float = 0.05
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")

    def update(self, value: float) -> tuple[float, float]:
        value = float(value)
        if not self.lower_bound <= value <= self.upper_bound:
            raise ValueError("value lies outside the declared bounds")
        self.count += 1
        self.total += value
        return self.interval()

    def interval(self) -> tuple[float, float]:
        if self.count == 0:
            return self.lower_bound, self.upper_bound
        mean = self.total / self.count
        width = self.upper_bound - self.lower_bound
        log_term = math.log((math.pi**2 * self.count**2) / (3.0 * self.delta))
        radius = width * math.sqrt(log_term / (2.0 * self.count))
        return (
            max(self.lower_bound, mean - radius),
            min(self.upper_bound, mean + radius),
        )
