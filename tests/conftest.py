import pytest

# NB: `agent` is imported inside the fixtures, not at module level. conftest is
# loaded before the test module, and test_agent.py seeds AGENTS_SOCIETY_API_KEY
# and friends into os.environ *before* importing agent — importing it here first
# would freeze the real (empty) environment into agent's module globals.


@pytest.fixture(autouse=True)
def _reset_llm_provider_state():
    """call_llm remembers providers that failed account-side, so a later call
    in the same process skips them. That memo is process-scoped by design —
    reset it between tests so one failing-provider test can't mute the next.
    """
    import agent

    agent._DEAD_PROVIDERS.clear()
    yield
    agent._DEAD_PROVIDERS.clear()


@pytest.fixture(autouse=True)
def _no_stray_fallback_slots(monkeypatch):
    """A real LLM_ENDPOINT_2 in the developer's shell would silently lengthen
    the chain and change call counts. Tests configure their own slots."""
    import agent

    for slot in range(2, agent.MAX_LLM_PROVIDERS + 1):
        for name in ("LLM_ENDPOINT", "LLM_MODEL", "LLM_API_KEY"):
            monkeypatch.delenv(f"{name}_{slot}", raising=False)
