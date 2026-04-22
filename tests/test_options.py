"""Tests for Options framework (temporal abstraction / chunked behavior)."""

import numpy as np

from fpi.agent.options import Option, OptionDiscovery, OptionExecutor


class TestOption:
    def test_creation(self):
        o = Option(option_id=0, action_sequence=(1, 2, 0),
                   initiation_pattern_id=5, expected_valence=0.3)
        assert o.option_id == 0
        assert o.action_sequence == (1, 2, 0)

    def test_confidence_zero_initially(self):
        o = Option(option_id=0, action_sequence=(1,),
                   initiation_pattern_id=0, expected_valence=0.1)
        assert o.confidence == 0.0

    def test_confidence_after_execution(self):
        o = Option(option_id=0, action_sequence=(1,),
                   initiation_pattern_id=0, expected_valence=0.1,
                   execution_count=10, success_count=7)
        assert o.confidence == 0.7


class TestOptionDiscovery:
    def test_discovers_repeated_sequence(self):
        disc = OptionDiscovery(min_repetitions=3, max_option_length=3)
        # Simulate: from pattern 5, action sequence (2, 2) with positive outcome
        # Discovery requires subsequences of length >= 2
        for _ in range(5):
            disc.observe(5, 2, 0.3)
            disc.discover()  # Each call counts multi-action subsequences

        # After 5 discovers, should have counted the (2, 2) sequence from pattern 5
        assert disc._sequence_counts.get((5, (2, 2)), 0) >= 3

    def test_no_discovery_before_min_reps(self):
        disc = OptionDiscovery(min_repetitions=5, max_option_length=3)
        # Only 2 repetitions
        for _ in range(2):
            disc.observe(5, 1, 0.1)
            disc.observe(5, 2, 0.3)

        options = disc.discover()
        # Shouldn't have enough reps yet
        assert len(options) == 0

    def test_no_discovery_for_negative_sequences(self):
        disc = OptionDiscovery(min_repetitions=2, max_option_length=3)
        for _ in range(5):
            disc.observe(5, 1, -0.5)
            disc.observe(5, 2, -0.3)

        options = disc.discover()
        assert len(options) == 0

    def test_does_not_rediscover(self):
        disc = OptionDiscovery(min_repetitions=2, max_option_length=3)
        for _ in range(5):
            disc.observe(5, 1, 0.1)
            disc.observe(5, 2, 0.3)

        options1 = disc.discover()
        options2 = disc.discover()  # Same data → no new options
        # Second call shouldn't produce new options for same sequences
        new_ids = {o.option_id for o in options2} - {o.option_id for o in options1}
        assert len(new_ids) == 0


class TestOptionExecutor:
    def test_add_and_initiate(self):
        exe = OptionExecutor()
        o = Option(option_id=0, action_sequence=(1, 2, 0),
                   initiation_pattern_id=5, expected_valence=0.3,
                   execution_count=5, success_count=4)
        exe.add_option(o)
        result = exe.should_initiate(current_pattern_id=5, urgency=0.3)
        assert result is not None
        assert result.option_id == 0

    def test_follows_action_sequence(self):
        exe = OptionExecutor()
        o = Option(option_id=0, action_sequence=(1, 2, 0),
                   initiation_pattern_id=5, expected_valence=0.3,
                   execution_count=5, success_count=4)
        exe.add_option(o)
        exe.should_initiate(current_pattern_id=5, urgency=0.3)

        assert exe.next_action() == 1
        assert exe.next_action() == 2
        assert exe.next_action() == 0
        # After sequence completes, should auto-terminate
        assert exe.next_action() is None
        assert exe.active_option is None

    def test_no_initiate_when_active(self):
        exe = OptionExecutor()
        o = Option(option_id=0, action_sequence=(1, 2),
                   initiation_pattern_id=5, expected_valence=0.3,
                   execution_count=5, success_count=4)
        exe.add_option(o)
        exe.should_initiate(current_pattern_id=5, urgency=0.3)
        # Can't initiate another while one is active
        result = exe.should_initiate(current_pattern_id=5, urgency=0.3)
        assert result is None

    def test_no_initiate_high_urgency(self):
        exe = OptionExecutor()
        o = Option(option_id=0, action_sequence=(1, 2),
                   initiation_pattern_id=5, expected_valence=0.3,
                   execution_count=5, success_count=4)
        exe.add_option(o)
        result = exe.should_initiate(current_pattern_id=5, urgency=0.9)
        assert result is None

    def test_terminate_success_updates_count(self):
        exe = OptionExecutor()
        o = Option(option_id=0, action_sequence=(1,),
                   initiation_pattern_id=5, expected_valence=0.3,
                   execution_count=5, success_count=3)
        exe.add_option(o)
        exe.should_initiate(current_pattern_id=5, urgency=0.3)
        exe.next_action()  # Execute the action
        exe.next_action()  # Auto-terminates with success
        assert o.success_count == 4  # Incremented

    def test_terminate_failure(self):
        exe = OptionExecutor()
        o = Option(option_id=0, action_sequence=(1, 2, 0),
                   initiation_pattern_id=5, expected_valence=0.3,
                   execution_count=5, success_count=3)
        exe.add_option(o)
        exe.should_initiate(current_pattern_id=5, urgency=0.3)
        exe.next_action()  # action 1
        exe.terminate(success=False)  # Early termination
        assert o.success_count == 3  # Not incremented
        assert exe.active_option is None

    def test_no_initiate_wrong_pattern(self):
        exe = OptionExecutor()
        o = Option(option_id=0, action_sequence=(1,),
                   initiation_pattern_id=5, expected_valence=0.3,
                   execution_count=5, success_count=4)
        exe.add_option(o)
        result = exe.should_initiate(current_pattern_id=99, urgency=0.3)
        assert result is None

    def test_capacity_eviction(self):
        exe = OptionExecutor(max_options=2)
        o1 = Option(option_id=0, action_sequence=(1,),
                    initiation_pattern_id=0, expected_valence=0.1,
                    execution_count=10, success_count=9)
        o2 = Option(option_id=1, action_sequence=(2,),
                    initiation_pattern_id=1, expected_valence=0.1,
                    execution_count=10, success_count=2)
        o3 = Option(option_id=2, action_sequence=(0,),
                    initiation_pattern_id=2, expected_valence=0.1,
                    execution_count=10, success_count=8)
        exe.add_option(o1)
        exe.add_option(o2)
        exe.add_option(o3)
        assert exe.option_count == 2
        # o2 (lowest confidence 0.2) should have been evicted
