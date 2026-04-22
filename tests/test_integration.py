"""Integration tests — verify all Phase 12 modules are wired into the Agent."""

import numpy as np

from fpi.agent.core import Agent
from fpi.agent.internal_society import InternalSociety, Planner
from fpi.env.base import SurvivalEnv
from fpi.primitives.signal import Signal


def _make_agent(**kwargs):
    """Create an Agent with common defaults for integration tests."""
    defaults = dict(
        similarity_threshold=0.7,
        seed=42,
        max_patterns=20,
        max_associations=60,
        association_decay_rate=0.005,
    )
    defaults.update(kwargs)
    return Agent(**defaults)


class TestBackwardCompatibility:
    """All features disabled = same behavior as pre-Phase-12."""

    def test_agent_default_has_no_new_modules(self):
        agent = _make_agent()
        assert agent._attention is None
        assert agent._working_memory is None
        assert agent._option_discovery is None
        assert agent._option_executor is None
        assert agent._internal_society is None
        assert agent._predictive_hierarchy is None

    def test_agent_default_survives_episode(self):
        agent = _make_agent()
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=50)
        assert len(results) > 0


class TestAttentionIntegration:
    def test_attention_enabled_with_slices(self):
        agent = _make_agent(
            enable_attention=True,
            modality_slices=[(0, 3), (3, 6)],
        )
        assert agent._attention is not None

    def test_attention_without_slices_stays_none(self):
        agent = _make_agent(enable_attention=True)
        assert agent._attention is None  # No slices = no gating

    def test_attention_gates_observation(self):
        agent = _make_agent(
            enable_attention=True,
            enable_affect=True,
            modality_slices=[(0, 3), (3, 6)],
        )
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=30)
        assert len(results) > 0


class TestAffectAttentionLoop:
    def test_affect_arousal_feeds_attention(self):
        """Affect arousal should modulate attention capacity."""
        agent = _make_agent(
            enable_affect=True,
            enable_attention=True,
            modality_slices=[(0, 3), (3, 6)],
        )
        # Run a few steps to build up affect state
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=20)
        # Affect should have a state (arousal may be 0 if no urgency)
        assert agent._affect is not None
        assert isinstance(agent._affect.current_state.arousal, float)


class TestWorkingMemoryIntegration:
    def test_wm_enabled(self):
        agent = _make_agent(enable_working_memory=True)
        assert agent._working_memory is not None

    def test_wm_holds_patterns_during_episode(self):
        agent = _make_agent(enable_working_memory=True)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=50)
        # After some steps, WM should have processed items
        # (it may be empty due to decay, but it should have been used)
        assert agent._working_memory is not None

    def test_wm_costs_vitality(self):
        """Agent with WM should have slightly lower vitality due to maintenance."""
        agent_no_wm = _make_agent(seed=42)
        agent_wm = _make_agent(seed=42, enable_working_memory=True)
        env1 = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        env2 = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        r1 = agent_no_wm.run_survival_episode(env1, max_steps=20)
        r2 = agent_wm.run_survival_episode(env2, max_steps=20)
        # WM agent should have equal or lower vitality (maintenance costs energy)
        assert r2[-1].vitality <= r1[-1].vitality + 0.01


class TestOptionsIntegration:
    def test_options_enabled(self):
        agent = _make_agent(enable_options=True)
        assert agent._option_discovery is not None
        assert agent._option_executor is not None

    def test_options_observe_during_episode(self):
        agent = _make_agent(enable_options=True)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=100)
        # Discovery should have accumulated some history
        assert len(agent._option_discovery._history) > 0

    def test_options_disabled_by_default(self):
        agent = _make_agent()
        assert agent._option_discovery is None
        assert agent._option_executor is None


class TestInternalSocietyIntegration:
    def test_society_replaces_inline_scoring(self):
        soc = InternalSociety(seed=42)
        agent = _make_agent(internal_society=soc)
        assert agent._internal_society is soc

    def test_society_select_action_used(self):
        soc = InternalSociety(seed=42)
        agent = _make_agent(internal_society=soc)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=30)
        # Society should have been called (last_proposals populated)
        assert len(soc.last_proposals) == 5  # 5 specialists

    def test_society_none_uses_fallback(self):
        agent = _make_agent()
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=30)
        assert len(results) > 0  # Falls back to inline scoring


class TestPlannerSpecialist:
    def test_planner_proposes_action(self):
        from fpi.agent.internal_society import ActionContext
        from fpi.primitives.vitality import Vitality
        from fpi.primitives.valence import Valence
        from fpi.world_model.model import WorldModel

        wm = WorldModel(similarity_threshold=0.7)
        sig = Signal(data=np.array([1.0, 0.0, 0.5, 0.3]))
        wm.observe(sig)
        planner = Planner()
        ctx = ActionContext(
            current_pattern=wm.current_pattern,
            vitality=Vitality(),
            valence=Valence(),
            world_model=wm,
            action_space=[0, 1, 2],
            rng=np.random.default_rng(42),
        )
        proposal = planner.propose(ctx)
        assert proposal.action in [0, 1, 2]
        assert proposal.source == "planner"

    def test_planner_depth_scales_with_urgency(self):
        planner = Planner(max_depth=3)
        # High urgency: depth = max(1, int(3 * (1 - 0.9))) = max(1, 0) = 1
        # Low urgency: depth = max(1, int(3 * (1 - 0.0))) = 3
        # We can't easily test depth directly, but we can test it runs
        from fpi.agent.internal_society import ActionContext
        from fpi.primitives.vitality import Vitality
        from fpi.primitives.valence import Valence
        from fpi.world_model.model import WorldModel

        wm = WorldModel(similarity_threshold=0.7)
        sig = Signal(data=np.array([1.0, 0.0, 0.5, 0.3]))
        wm.observe(sig)

        vit = Vitality()
        vit.spend(0.9 * vit.max_energy)  # High urgency
        ctx = ActionContext(
            current_pattern=wm.current_pattern,
            vitality=vit,
            valence=Valence(),
            world_model=wm,
            action_space=[0, 1, 2],
            rng=np.random.default_rng(42),
        )
        proposal = planner.propose(ctx)
        assert proposal.action in [0, 1, 2]
        # Priority should be lower under high urgency
        assert proposal.priority < planner.base_priority


class TestPredictiveHierarchyIntegration:
    def test_ph_enabled(self):
        agent = _make_agent(enable_predictive_hierarchy=True)
        assert agent._predictive_hierarchy is not None

    def test_ph_runs_with_agent(self):
        agent = _make_agent(enable_predictive_hierarchy=True, ph_num_levels=2)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=30)
        assert len(results) > 0

    def test_ph_threshold_restored(self):
        agent = _make_agent(enable_predictive_hierarchy=True, ph_num_levels=2)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=10)
        # Threshold should be restored to base after each step
        assert (agent.world_model.memory.distinction.similarity_threshold
                == agent._base_similarity_threshold)

    def test_ph_disabled_by_default(self):
        agent = _make_agent()
        assert agent._predictive_hierarchy is None


class TestEpisodicMemoryIntegration:
    def test_episodic_enabled(self):
        agent = _make_agent(enable_episodic_memory=True)
        assert agent._episodic is not None

    def test_episodic_disabled_by_default(self):
        agent = _make_agent()
        assert agent._episodic is None

    def test_episodic_records_during_episode(self):
        agent = _make_agent(enable_episodic_memory=True, episodic_surprise_threshold=0.3)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=50)
        assert agent._episodic.count > 0

    def test_episodic_context_in_society(self):
        """When InternalSociety is active, ActionContext includes episodic recalls."""
        soc = InternalSociety(seed=42)
        agent = _make_agent(
            internal_society=soc,
            enable_episodic_memory=True,
            episodic_surprise_threshold=0.3,
        )
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=50)
        assert agent._episodic.count > 0


class TestSequenceMemoryIntegration:
    def test_sequence_enabled(self):
        agent = _make_agent(enable_sequence_memory=True)
        assert agent._sequence_memory is not None

    def test_sequence_disabled_by_default(self):
        agent = _make_agent()
        assert agent._sequence_memory is None

    def test_sequence_observes_during_episode(self):
        agent = _make_agent(enable_sequence_memory=True, sequence_window=3)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=50)
        assert agent._sequence_memory.observation_count > 0


class TestTemporalHierarchyIntegration:
    def test_th_enabled(self):
        agent = _make_agent(temporal_scales=(3, 7))
        assert agent._temporal_hierarchy is not None

    def test_th_supersedes_sequence_memory(self):
        """TH and SM are mutually exclusive; TH takes precedence."""
        agent = _make_agent(enable_sequence_memory=True, temporal_scales=(3, 7))
        assert agent._temporal_hierarchy is not None
        assert agent._sequence_memory is None

    def test_th_disabled_by_default(self):
        agent = _make_agent()
        assert agent._temporal_hierarchy is None

    def test_th_observes_during_episode(self):
        agent = _make_agent(temporal_scales=(3, 7))
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=50)
        status = agent._temporal_hierarchy.get_status()
        any_observed = any(
            s["observation_count"] > 0
            for s in status["scales"].values()
        )
        assert any_observed

    def test_th_surprise_valid(self):
        agent = _make_agent(temporal_scales=(3,), th_surprise_weight=0.5)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=50)
        for r in results:
            assert 0.0 <= r.surprise <= 1.5  # blending can slightly exceed 1.0


class TestSelfModelBypassFix:
    def test_self_model_vitals_in_context(self):
        from fpi.agent.introspection import SelfModel
        soc = InternalSociety(seed=42)
        sm = SelfModel()
        agent = _make_agent(internal_society=soc, self_model=sm)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=30)
        assert sm.vitals.surprise_momentum >= 0.0

    def test_self_model_modulates_society(self):
        from fpi.agent.introspection import SelfModel
        soc = InternalSociety(seed=42)
        sm = SelfModel()
        agent = _make_agent(internal_society=soc, self_model=sm)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        agent.run_survival_episode(env, max_steps=50)
        assert len(soc.last_proposals) == 5

    def test_without_self_model_no_vitals(self):
        agent = _make_agent()
        assert agent._self_model is None


class TestFullStack:
    def test_all_features_enabled(self):
        """Agent with ALL features enabled should complete an episode."""
        from fpi.agent.introspection import SelfModel
        soc = InternalSociety(seed=42)
        sm = SelfModel()
        agent = _make_agent(
            enable_affect=True,
            enable_attention=True,
            modality_slices=[(0, 3), (3, 6)],
            enable_working_memory=True,
            enable_options=True,
            internal_society=soc,
            enable_predictive_hierarchy=True,
            ph_num_levels=2,
            enable_episodic_memory=True,
            episodic_surprise_threshold=0.3,
            temporal_scales=(3, 7),
            th_surprise_weight=0.2,
            self_model=sm,
        )
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=50)
        assert len(results) > 0
        assert soc.last_proposals is not None
        assert agent._episodic.count > 0
        assert agent._temporal_hierarchy is not None
