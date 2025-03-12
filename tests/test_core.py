import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from themefinder import (
    find_themes,
    sentiment_analysis,
    theme_generation,
    theme_condensation,
    theme_refinement,
    theme_target_alignment,
    theme_mapping,
)


@pytest.mark.asyncio
async def test_sentiment_analysis(mock_llm, sample_df):
    """Test sentiment analysis with mocked LLM responses."""
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "sentiment": "positive"},
                    {"response_id": 2, "sentiment": "negative"},
                ]
            }
        )
    )
    result = await sentiment_analysis(
        sample_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "sentiment" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_generation(mock_llm, sample_df):
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
        sample_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "themes" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_condensation(mock_llm):
    """Test theme condensation with mocked LLM responses."""
    initial_df = pd.DataFrame({"theme": [f"theme{i}" for i in range(1, 6)]})
    mock_llm.ainvoke.side_effect = [
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}, {"theme": "B"}, {"theme": "C"}]})),
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}, {"theme": "B"}, {"theme": "C"}]})),
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
async def test_theme_mapping(mock_llm, sample_df):
    """Test theme mapping with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "reason": "reason1", "label": ["label1"]},
                    {"response_id": 2, "reason": "reason2", "label": ["label2"]},
                ]
            }
        )
    )
    result = await theme_mapping(
        sample_df, mock_llm, question="test question", refined_themes_df=refined_df, batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "response_id" in result.columns
    assert "reason" in result.columns
    assert "label" in result.columns
    assert mock_llm.ainvoke.call_count == 1

@pytest.mark.asyncio
async def test_sentiment_analysis(mock_llm, sample_df):
    """Test sentiment analysis with mocked LLM responses."""
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "sentiment": "positive"},
                    {"response_id": 2, "sentiment": "negative"},
                ]
            }
        )
    )
    result = await sentiment_analysis(
        sample_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "sentiment" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_generation(mock_llm, sample_df):
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
        sample_df, mock_llm, question="test question", batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "themes" in result.columns
    assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_theme_condensation(mock_llm):
    """Test theme condensation with mocked LLM responses."""
    initial_df = pd.DataFrame({"theme": [f"theme{i}" for i in range(1, 6)]})
    mock_llm.ainvoke.side_effect = [
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}, {"theme": "B"}, {"theme": "C"}]})),
        MagicMock(content=json.dumps({"responses": [{"theme": "A"}, {"theme": "B"}, {"theme": "C"}]})),
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
async def test_theme_mapping(mock_llm, sample_df):
    """Test theme mapping with mocked LLM responses."""
    refined_df = pd.DataFrame({"topic_id": ["1", "2"], "topic": ["theme1", "theme2"]})
    mock_llm.ainvoke.return_value = MagicMock(
        content=json.dumps(
            {
                "responses": [
                    {"response_id": 1, "reason": "reason1", "label": ["label1"]},
                    {"response_id": 2, "reason": "reason2", "label": ["label2"]},
                ]
            }
        )
    )
    result = await theme_mapping(
        sample_df, mock_llm, question="test question", refined_themes_df=refined_df, batch_size=2
    )
    assert isinstance(result, pd.DataFrame)
    assert "response_id" in result.columns
    assert "reason" in result.columns
    assert "label" in result.columns
    assert mock_llm.ainvoke.call_count == 1

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
    result = await find_themes(
        sample_df, mock_llm, question="test question", target_n_themes=2
    )
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in [
            "sentiment",
            "themes",
            "condensed_themes",
            "mapping",
            "question",
            "refined_themes",
        ]
    )
    assert mock_llm.ainvoke.call_count == 7


@pytest.mark.asyncio
async def test_theme_condensation(monkeypatch):
    """
    Test the while loop behavior in theme_condensation.

    The test creates an initial DataFrame with 5 rows and uses a batch size of 2.
    It patches batch_and_run with a dummy function that:
      - On the first call, returns a DataFrame with 3 rows (reducing the row count).
      - On the second call, returns a DataFrame with the same 3 rows (causing the loop to break).
      - On the final call (after the loop), returns a DataFrame with 2 rows.
    """
    initial_df = pd.DataFrame({"theme": [f"theme{i}" for i in range(1, 6)]})
    df_first = pd.DataFrame({"theme": ["A", "B", "C"]})
    df_second = pd.DataFrame({"theme": ["A", "B", "C"]})
    df_final = pd.DataFrame({"theme": ["A", "B"]})
    dummy_outputs = [df_first, df_second, df_final]

    call_count = 0

    async def dummy_batch_and_run(
        themes_df, prompt_template, llm, batch_size, question, system_prompt, **kwargs
    ):
        nonlocal call_count
        call_count += 1
        return dummy_outputs.pop(0)

    monkeypatch.setitem(
        theme_condensation.__globals__, "batch_and_run", dummy_batch_and_run
    )
    dummy_llm = MagicMock()

    await theme_condensation(
        themes_df=initial_df.copy(),
        llm=dummy_llm,
        question="test question",
        batch_size=2,
    )
    assert call_count == 3, "batch_and_run should have been called 3 times"
