from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main


@pytest.mark.parametrize(
    ("text", "expected"),
    [("  PRESSÃO e Temperatura  ", "pressao e temperatura"), ("Março", "marco"), ("", "")],
)
def test_normalize_text(text, expected):
    assert main.normalize_text(text) == expected


def test_extract_entities_interprets_date_priority_and_values():
    entities = main.extract_entities("Urgente: 125,5 kg e 42 kWh em 15/03/2026")
    assert entities["priority"] == "high"
    assert entities["waste_processed"] == 125.5
    assert entities["energy_generated"] == 42
    assert (entities["day"], entities["month"], entities["year"]) == (15, 2, 2026)


@pytest.mark.parametrize(
    ("text", "topic"),
    [("pressão alta", "pressao"), ("cheiro de gás sulfídrico", "h2s"), ("acidez do biodigestor", "ph")],
)
def test_extract_operational_topics(text, topic):
    assert main.extract_entities(text)["operational_topic"] == topic


def test_transcribe_file_joins_segments_without_loading_whisper(monkeypatch):
    model = Mock()
    model.transcribe.return_value = (
        [SimpleNamespace(text=" Olá "), SimpleNamespace(text=" biodigestor ")],
        SimpleNamespace(language="pt", duration=1.234),
    )
    monkeypatch.setattr(main, "get_whisper_model", lambda: model)

    result = main.transcribe_file("audio.wav")

    assert result == {"text": "Olá biodigestor", "language": "pt", "duration": 1.23, "model": main.WHISPER_MODEL}
    model.transcribe.assert_called_once()
    assert model.transcribe.call_args.args == ("audio.wav",)
    assert model.transcribe.call_args.kwargs["language"] == "pt"
