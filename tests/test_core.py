import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from themefinder import (
    sentiment_analysis,
    theme_condensation,
    theme_generation,
    theme_mapping,
    theme_refinement,
)
from themefinder.core import theme_target_alignment


@pytest.mark.asyncio
async def test_sentiment_analysis(mock_llm, sample_df):
    """Test sentiment analysis with mocked LLM responses."""
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "position": "positive"},
                    {"response_id": 2, "position": "negative"},
                ]
            }
        )
    )
    result = await sentiment_analysis(
        sample_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "position" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_generation(mock_llm, sample_sentiment_df):
    """Test theme generation with mocked LLM responses."""
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "themes": ["theme1", "theme2"]},
                    {"response_id": 2, "themes": ["theme3", "theme4"]},
                ]
            }
        )
    )
    result = await theme_generation(
        sample_sentiment_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "themes" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_condensation(mock_llm):
    """Test theme condensation with mocked LLM responses."""
    initial_df = pd.DataFrame({"theme": [f"theme{i}" for i in range(1, 6)]})
    mock_llm.ainvoke.side_effect = [
        MagicMock(
            content=json.dumps(
                {"responses": [{"theme": "A"}, {"theme": "B"}, {"theme": "C"}]}
            )
        ),
        MagicMock(
            content=json.dumps(
                {"responses": [{"theme": "A"}, {"theme": "B"}, {"theme": "C"}]}
            )
        ),
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}, {"theme": "B"}]})),
    ]
    result = await theme_condensation(
        initial_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "theme" in result.columns
    assert mock_llm.ainvoke.call_count == 3


@pytest.mark.asyncio
async def test_theme_refinement(mock_llm):
    """Test theme refinement with mocked LLM responses."""
    condensed_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"topic_id": "1", "topic": "refined_theme1"},
                    {"topic_id": "2", "topic": "refined_theme2"},
                ]
            }
        )
    )
    result = await theme_refinement(
        condensed_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "1" in result.columns
    assert "2" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_target_alignment(mock_llm):
    """Test theme target alignment with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"topic_id": "1", "topic": "aligned_theme1"},
                    {"topic_id": "2", "topic": "aligned_theme2"},
                ]
            }
        )
    )
    result = await theme_target_alignment(
        refined_df, mock_llm, question="test question", target_n_themes=2, batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "1" in result.columns
    assert "2" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_mapping(mock_llm, sample_sentiment_df):
    """Test theme mapping with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "reason": ["reason1"], "label": ["label1"]},
                    {"response_id": 2, "reason": ["reason2"], "label": ["label2"]},
                ]
            }
        )
    )
    result = await theme_mapping(
        sample_sentiment_df,
        mock_llm,
        question="test question",
        refined_themes_df=refined_df,
        batch_size=2,
    )
    assert isinstance(result, pd.DataFrame)
    assert "response_id" in result.columns
    assert "reason" in result.columns
    assert "label" in result.columns
    assert mock_llm.ainvoke.call_count == 1
