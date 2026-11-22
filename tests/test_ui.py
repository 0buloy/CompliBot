import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_ui_renders_and_answers(tmp_path, monkeypatch, index, settings):
    monkeypatch.setenv("COMPLIBOT_INDEX_DIR", str(settings.index_dir))
    monkeypatch.setenv("COMPLIBOT_LLM_PROVIDER", "fake")
    monkeypatch.delenv("COMPLIBOT_API_URL", raising=False)
    at = AppTest.from_file("../ui/app.py", default_timeout=30).run()
    assert not at.exception
    at.text_area(key="question").input("How many business days to stop using data after consent is withdrawn?").run()
    submit = next(b for b in at.button if b.label == "Submit")  # enabled once there is a question
    submit.click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "10 business days" in page
    assert any("s.2.3" in e.label for e in at.expander)
