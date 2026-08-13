"""Qt stylesheet built from semantic design tokens."""

# ruff: noqa: E501 -- one selector per line keeps the generated QSS auditable.

from app.design_system.tokens import COLORS, RADIUS, SPACE

APP_STYLESHEET = f"""
QMainWindow, QWidget {{ background: {COLORS["canvas"]}; color: {COLORS["text"]}; }}
QFrame#card {{ background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; border-radius: {RADIUS["md"]}px; }}
QFrame#hint {{ background: {COLORS["success_surface"]}; border: 1px solid {COLORS["success_border"]}; border-radius: {RADIUS["md"]}px; }}
QFrame#rail {{ background: {COLORS["rail"]}; border-right: 1px solid {COLORS["border"]}; }}
QLabel#hintText {{ color: #7EE2B8; }}
QLabel#h1 {{ font-size: 16px; font-weight: 600; }}
QLabel#big {{ font-size: 20px; font-weight: 600; }}
QLabel#mono {{ font-family: Consolas, "Courier New", monospace; color: #9DA7B3; }}
QLabel#muted {{ color: {COLORS["text_muted"]}; }}
QLabel#err {{ color: {COLORS["danger"]}; }}
QLabel#step {{ padding: {SPACE["sm"]}px 10px; border-radius: {RADIUS["sm"]}px; color: {COLORS["text_muted"]}; }}
QLabel#stepActive {{ padding: {SPACE["sm"]}px 10px; border-radius: {RADIUS["sm"]}px; background: {COLORS["accent_strong"]}; color: white; }}
QLabel#stepDone {{ padding: {SPACE["sm"]}px 10px; border-radius: {RADIUS["sm"]}px; color: {COLORS["success"]}; }}
QPushButton {{ background: {COLORS["accent_strong"]}; color: white; border: 0; padding: {SPACE["sm"]}px {SPACE["md"]}px; border-radius: {RADIUS["sm"]}px; }}
QPushButton:hover {{ background: {COLORS["accent"]}; }}
QPushButton:disabled {{ background: {COLORS["disabled"]}; color: {COLORS["text_muted"]}; }}
QSpinBox {{ background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; border-radius: {RADIUS["sm"]}px; padding: {SPACE["xs"]}px; }}
"""
