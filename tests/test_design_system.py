"""Keep Qt and chart rendering on the shared semantic token contract."""

from app.design_system.theme import APP_STYLESHEET
from app.design_system.tokens import COLORS


def test_required_semantic_tokens_exist() -> None:
    required = {"canvas", "surface", "border", "text", "accent", "success", "warning", "danger"}
    assert required <= COLORS.keys()


def test_stylesheet_uses_semantic_tokens() -> None:
    assert COLORS["canvas"] in APP_STYLESHEET
    assert COLORS["accent_strong"] in APP_STYLESHEET
    assert COLORS["danger"] in APP_STYLESHEET
