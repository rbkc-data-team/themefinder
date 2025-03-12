from unittest.mock import AsyncMock

import pandas as pd
import pytest


@pytest.fixture()
def mock_llm():
    mock = AsyncMock()
    return mock


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"response_id": [1, 2], "text": ["response1", "response2"]})


@pytest.fixture()
def sample_sentiment_df():
    return pd.DataFrame(
        {
            "response_id": [1, 2],
            "text": ["response1", "response2"],
            "position": ["positive", "negative"],
        }
    )
