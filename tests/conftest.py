import pytest

from sentinel_semantic.router import SemanticRouter


@pytest.fixture(scope="session")
def router():
    """Deterministic keyword engine so tests never depend on a model download."""
    return SemanticRouter(engine="keyword")
