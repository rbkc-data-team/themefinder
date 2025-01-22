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


@mock.patch("tenacity.nap.time.sleep", MagicMock())
@pytest.mark.asyncio()
async def test_id_integrity_failure(mock_llm):
    """
    Test the behavior when LLM returns responses with mismatched IDs.
    Verifies that the system recovers by splitting the batch and retrying
    when response IDs don't match the input DataFrame.
    """
    input_dataframe = pd.DataFrame(
        {"response_id": [1, 2], "text": ["response1", "response2"]}
    )

    mock_llm.ainvoke.side_effect = [
        MagicMock(
            # note different ID in response
            content='{"responses": [{"response_id": 3, "position": "agreement", "text": "response1"}, {"response_id": 2, "position": "disagreement", "text": "response2"}]}'
        ),
        MagicMock(
            # split: first handle response 1...
            content='{"responses": [{"response_id": 1, "position": "agreement", "text": "response1"}]}'
        ),
        MagicMock(
            # split: ...then handle response 2
            content='{"responses": [{"response_id": 2, "position": "agreement", "text": "response1"}]}'
        ),
    ]

    result = await sentiment_analysis(
        input_dataframe, mock_llm, question="doesn't matter"
    )
    # we got something back
    assert isinstance(result, pd.DataFrame)
    assert list(result["response_id"]) == [1, 2]
    assert mock_llm.ainvoke.call_count == 3


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
