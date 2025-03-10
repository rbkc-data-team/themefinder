from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest

from themefinder import sentiment_analysis
from themefinder.llm_batch_processor import process_llm_responses


async def test_process_llm_responses_with_clashing_types():
    """
    Test that process_llm_responses handles type mismatches between response IDs.
    Verifies that string response IDs from LLM responses are correctly matched
    with integer response IDs in the original DataFrame.
    """
    responses = pd.DataFrame({"response_id": [1], "text": ["response1"]})
    processed = process_llm_responses(
        # llm gives us a str response_id but the original response_id is an int
        [{"responses": [{"response_id": "1", "llm_contribution": "banana"}]}],
        responses,
    )
    assert list(processed["response_id"]) == [1]
    assert list(processed["llm_contribution"]) == ["banana"]

@pytest.mark.asyncio()
async def test_retries(mock_llm, sample_df):
    """
    Test the retry mechanism when LLM calls fail.
    Verifies that the system properly retries after an exception
    and successfully processes the responses on subsequent attempts.
    """
    exception = Exception("Rate limited!")
    mock_llm.ainvoke.side_effect = [
        exception,
        MagicMock(
            content='{"responses": [{"response_id": 1, "position": "agreement", "text": "response1"}, {"response_id": 2, "position": "disagreement", "text": "response2"}]}'
        ),
    ]
    result = await sentiment_analysis(sample_df, mock_llm, question="doesn't matter")
    # we got something back
    assert isinstance(result, pd.DataFrame)
    # we hit the llm twice
    assert mock_llm.ainvoke.call_count == 2
