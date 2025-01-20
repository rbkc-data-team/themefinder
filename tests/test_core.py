import json
from unittest.mock import MagicMock

import pytest

from themefinder import find_themes


@pytest.mark.asyncio()
async def test_find_themes(mock_llm, sample_df):
    """Test the complete theme finding pipeline with mocked LLM responses."""
    mock_llm.ainvoke.side_effect = [
        MagicMock(
            content='{"responses": [{"response_id": 1, "position": "agreement", "text": "response1"}, {"response_id": 2, "position": "disagreement", "text": "response2"}]}'
        ),
        MagicMock(content='{"responses": [{"themes": ["theme1", "theme2"]}]}'),
        MagicMock(content='{"responses": [{"themes": ["theme3", "theme4"]}]}'),
        MagicMock(
            content='{"responses": [{"condensed_themes": ["main_theme1", "main_theme2"]}]}'
        ),
        MagicMock(content='{"responses": [{"topic_id": "label1", "topic": "desc1"}]}'),
        MagicMock(
            content=json.dumps(
                {
                    "responses": [
                        {
                            "response_id": 1,
                            "reason": "reason1",
                            "label": ["label1"],
                        },
                        {
                            "response_id": 2,
                            "reason": "reason2",
                            "label": ["label2"],
                        },
                    ]
                }
            )
        ),
    ]
    result = await find_themes(sample_df, mock_llm, expanded_question="test question")
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in [
            "sentiment",
            "topics",
            "condensed_topics",
            "mapping",
            "expanded_question",
            "refined_topics",
        ]
    )
    assert mock_llm.ainvoke.call_count == 6
