"""Options — learned multi-step action sequences (temporal abstraction).

Implements the Options framework (Sutton, Precup, Singh 1999): an option
is a temporally extended action with:
- Initiation condition: which pattern triggers it
- Internal policy: the action sequence to execute
- Termination condition: when to stop

Options are auto-discovered from repeated successful action sequences,
not hand-designed. When the agent repeatedly executes A→B→C and gets
positive valence, it chunks this into an option "do-ABC-from-pattern-X."

This is how habits form: frequently successful behaviors get chunked
into single units, freeing working memory for higher-level planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Option:
    """A learned multi-step action sequence (chunked behavior)."""

    option_id: int
    action_sequence: tuple[int, ...]
    initiation_pattern_id: int
    expected_valence: float
    execution_count: int = 0
    success_count: int = 0

    @property
    def confidence(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


@dataclass(slots=True)
class _ActionRecord:
    """Internal: one step in the agent's recent action history."""

    pattern_id: int
    action: int
    valence: float


class OptionDiscovery:
    """Discovers options from repeated successful action sequences.

    Watches the agent's recent (pattern, action, valence) triples.
    When a sequence of actions from a specific pattern repeatedly leads
    to positive cumulative valence, it becomes an Option.

    Args:
        min_repetitions: How many times a sequence must succeed to become an option.
        max_option_length: Maximum actions in a single option.
        history_length: How many recent actions to track.
    """

    def __init__(
        self,
        min_repetitions: int = 3,
        max_option_length: int = 5,
        history_length: int = 50,
    ) -> None:
        self._min_repetitions = min_repetitions
        self._max_option_length = max_option_length
        self._history_length = history_length
        self._history: list[_ActionRecord] = []
        self._next_id = 0
        # (initiation_pattern_id, action_tuple) → count of successes
        self._sequence_counts: dict[tuple[int, tuple[int, ...]], int] = {}
        self._known_options: set[tuple[int, tuple[int, ...]]] = set()

    def observe(self, pattern_id: int, action: int, valence: float) -> None:
        """Record a (pattern, action, outcome) triple."""
        self._history.append(_ActionRecord(pattern_id, action, valence))
        if len(self._history) > self._history_length:
            self._history.pop(0)

    def discover(self) -> list[Option]:
        """Check for repeated successful sequences. Returns newly discovered options."""
        if len(self._history) < 3:
            return []

        new_options: list[Option] = []

        # Scan for successful sub-sequences ending at recent positive outcomes
        for end_idx in range(2, len(self._history)):
            if self._history[end_idx].valence <= 0:
                continue

            # Try different sequence lengths
            for length in range(2, min(self._max_option_length + 1, end_idx + 1)):
                start_idx = end_idx - length + 1
                subseq = self._history[start_idx:end_idx + 1]

                init_pattern = subseq[0].pattern_id
                actions = tuple(r.action for r in subseq)
                total_valence = sum(r.valence for r in subseq)

                if total_valence <= 0:
                    continue

                key = (init_pattern, actions)
                if key in self._known_options:
                    continue

                self._sequence_counts[key] = self._sequence_counts.get(key, 0) + 1

                if self._sequence_counts[key] >= self._min_repetitions:
                    option = Option(
                        option_id=self._next_id,
                        action_sequence=actions,
                        initiation_pattern_id=init_pattern,
                        expected_valence=total_valence / length,
                    )
                    new_options.append(option)
                    self._known_options.add(key)
                    self._next_id += 1

        return new_options


class OptionExecutor:
    """Manages option execution within the agent's action loop.

    When an option is active, it overrides select_action() with the
    option's internal policy (its action sequence).

    Args:
        max_options: Maximum number of learned options to store.
    """

    def __init__(self, max_options: int = 20) -> None:
        self._options: list[Option] = []
        self._max_options = max_options
        self._active_option: Option | None = None
        self._step_in_option: int = 0

    @property
    def active_option(self) -> Option | None:
        return self._active_option

    @property
    def option_count(self) -> int:
        return len(self._options)

    def add_option(self, option: Option) -> None:
        """Register a new option. Evicts lowest-confidence if at capacity."""
        if len(self._options) >= self._max_options:
            weakest = min(self._options, key=lambda o: o.confidence)
            self._options.remove(weakest)
        self._options.append(option)

    def should_initiate(
        self, current_pattern_id: int, urgency: float,
    ) -> Option | None:
        """Check if any option should start from current state.

        Only initiates if not already executing an option. Prefers
        higher-confidence options. Won't initiate under high urgency
        (stick to primitive actions when desperate).
        """
        if self._active_option is not None:
            return None
        if urgency > 0.8:
            return None

        candidates = [
            o for o in self._options
            if o.initiation_pattern_id == current_pattern_id
            and o.confidence > 0.3
        ]
        if not candidates:
            return None

        # Pick highest confidence
        best = max(candidates, key=lambda o: o.confidence)
        self._active_option = best
        self._step_in_option = 0
        best.execution_count += 1
        return best

    def next_action(self) -> int | None:
        """Return next action from active option, or None if no option active."""
        if self._active_option is None:
            return None
        if self._step_in_option >= len(self._active_option.action_sequence):
            # Option completed
            self.terminate(success=True)
            return None
        action = self._active_option.action_sequence[self._step_in_option]
        self._step_in_option += 1
        return action

    def terminate(self, success: bool) -> None:
        """End the active option, update success/failure stats."""
        if self._active_option is not None and success:
            self._active_option.success_count += 1
        self._active_option = None
        self._step_in_option = 0
