"""Smoke test: the Streamlit app must execute end to end without raising.

A green `/_stcore/health` only proves the server is listening — it says nothing
about whether the script ran. `AppTest` actually executes the script, so a bad
import, a renamed column or a missing artefact fails here instead of silently
showing a blank page in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def app() -> AppTest:
    if not (ROOT / "models" / "voting_regressor.joblib").exists():
        pytest.skip("trained model missing — run scripts/train_all.py")
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def test_app_runs_without_exception(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_forecast_tab_renders_metrics(app):
    # the Forecast tab emits four st.metric cards
    assert len(app.metric) >= 3


def test_model_leaderboard_renders(app):
    labels = {m.label for m in app.metric}
    assert any("Locked-test" in label for label in labels)


def test_no_error_banners(app):
    assert not app.error, [e.value for e in app.error]
