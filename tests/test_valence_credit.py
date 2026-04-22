"""Tests for retroactive valence adjustment (temporal credit assignment)."""

from fpi.primitives.valence import Valence


class TestAdjustRetroactive:
    def test_negative_outcome_penalizes_predecessors(self) -> None:
        """Patterns preceding a bad outcome should get negative valence."""
        v = Valence()
        # Initialize some patterns with neutral valence
        for pid in [1, 2, 3]:
            v.update(pid, 0.0)

        # Bad outcome: penalize the chain [1, 2, 3]
        v.adjust_retroactive([1, 2, 3], outcome_delta=-1.0, decay=0.85, strength=0.1)

        # Most recent (3) gets strongest penalty, oldest (1) gets weakest
        assert v.get(3) < 0.0
        assert v.get(2) < 0.0
        assert v.get(1) < 0.0
        assert v.get(3) < v.get(2) < v.get(1)

    def test_positive_outcome_boosts_predecessors(self) -> None:
        """Patterns preceding a good outcome should get positive valence."""
        v = Valence()
        for pid in [10, 20, 30]:
            v.update(pid, 0.0)

        v.adjust_retroactive([10, 20, 30], outcome_delta=1.0, decay=0.85, strength=0.1)

        assert v.get(30) > 0.0
        assert v.get(20) > 0.0
        assert v.get(10) > 0.0
        assert v.get(30) > v.get(20) > v.get(10)

    def test_decay_reduces_effect_with_distance(self) -> None:
        """The further back a pattern is, the weaker the adjustment."""
        v = Valence()
        pattern_ids = [1, 2, 3, 4, 5]
        for pid in pattern_ids:
            v.update(pid, 0.0)

        v.adjust_retroactive(pattern_ids, outcome_delta=-1.0, decay=0.5, strength=1.0)

        # Most recent (5) gets full strength, each step halves
        vals = [abs(v.get(pid)) for pid in pattern_ids]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1], f"vals[{i}] should be < vals[{i+1}]"

    def test_empty_pattern_list(self) -> None:
        """Empty pattern list should be a no-op."""
        v = Valence()
        v.adjust_retroactive([], outcome_delta=-1.0)
        assert v.known_count == 0

    def test_zero_outcome_delta_no_change(self) -> None:
        """Zero outcome delta should not change valence."""
        v = Valence()
        v.update(1, 0.5)
        val_before = v.get(1)
        v.adjust_retroactive([1], outcome_delta=0.0)
        assert v.get(1) == val_before

    def test_creates_valence_for_unknown_patterns(self) -> None:
        """Retroactive adjustment should work even for patterns without prior valence."""
        v = Valence()
        v.adjust_retroactive([99], outcome_delta=-1.0, strength=0.1)
        # The update call inside adjust_retroactive creates the entry
        assert v.is_known(99)
        assert v.get(99) < 0.0

    def test_strength_scales_adjustment(self) -> None:
        """Higher strength should produce larger adjustments."""
        v1 = Valence()
        v2 = Valence()

        v1.adjust_retroactive([1], outcome_delta=-1.0, strength=0.1)
        v2.adjust_retroactive([1], outcome_delta=-1.0, strength=0.5)

        assert abs(v2.get(1)) > abs(v1.get(1))
