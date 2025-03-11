from unittest.mock import MagicMock

import pandas as pd
import pytest
import tiktoken

from themefinder import sentiment_analysis
from themefinder.llm_batch_processor import (
    batch_task_input_df,
    build_prompt,
    calculate_string_token_length,
    generate_prompts,
    process_llm_responses,
)


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


def test_empty_string(monkeypatch):
    fake_encoding = MagicMock()
    fake_encoding.encode.return_value = []
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: fake_encoding)

    token_length = calculate_string_token_length("", model="test-model")
    assert token_length == 0


def test_non_empty_string(monkeypatch):
    fake_encoding = MagicMock()
    fake_encoding.encode.side_effect = lambda text: text.split()
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: fake_encoding)

    token_length = calculate_string_token_length("hello world", model="test-model")

    assert token_length == 2


def test_calls_encoding_for_model(monkeypatch):
    fake_encoding = MagicMock()
    fake_encoding.encode.return_value = ["token1", "token2", "token3"]
    fake_encoding_for_model = MagicMock(return_value=fake_encoding)
    monkeypatch.setattr(tiktoken, "encoding_for_model", fake_encoding_for_model)

    token_length = calculate_string_token_length("any text", model="custom-model")
    fake_encoding_for_model.assert_called_once_with("custom-model")

    assert token_length == 3


# Define a dummy prompt template with a template attribute and a predictable format() method.
class DummyPromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        # For testing, simply return a string that includes the number of responses and any extra info.
        responses = kwargs.get("responses", [])
        extra = kwargs.get("extra", "")
        return f"Prompt with {len(responses)} responses. {extra}"


def test_build_prompt():
    data = {"response_id": [1, 2], "text": ["response1", "response2"]}
    df = pd.DataFrame(data)
    prompt_template = DummyPromptTemplate("dummy template")

    extra_info = "Extra context"
    result = build_prompt(prompt_template, df, extra=extra_info)

    expected_prompt = (
        f"Prompt with {len(df.to_dict(orient='records'))} responses. {extra_info}"
    )
    assert result.prompt_string == expected_prompt
    assert result.response_ids == ["1", "2"]


def test_build_prompt_with_mock_template():
    data = {"response_id": [101, 202], "text": ["foo", "bar"]}
    df = pd.DataFrame(data)

    prompt_template = MagicMock()
    prompt_template.format.return_value = "formatted prompt"

    result = build_prompt(prompt_template, df)

    prompt_template.format.assert_called_once_with(
        responses=df.to_dict(orient="records")
    )

    assert result.prompt_string == "formatted prompt"
    assert result.response_ids == ["101", "202"]


def dummy_token_length_low(input_text: str, model="gpt-4o"):
    """
    Simulate a tokenizer that always returns a low token count.
    This forces the branch where the sample batch is small enough,
    so we simply split by batch size.
    """
    return 10


def dummy_token_length_iterative(input_text: str, model="gpt-4o"):
    """
    Simulate a tokenizer that returns a high token count for multi-row batches.
    We detect multiple rows by the presence of the sequence "},{"
    in the JSON string. For single-row JSON (which normally lacks "},{")
    return a low token count.
    """
    if "},{" in input_text:
        return 100  # High count to force iterative row-by-row splitting.
    return 10


def test_batch_task_input_df_row_based(monkeypatch):
    """
    Test that when the sample batch's token count is low (<= allowed_tokens),
    the DataFrame is simply split by row count.
    """
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_low,
    )
    # Create a DataFrame with 5 rows.
    df = pd.DataFrame(
        {"response_id": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]}
    )
    batch_size = 2
    allowed_tokens = 100  # High enough so that sample_batch token count (10 * number of rows in sample)
    # is below the allowed_tokens.

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key=None)

    # Expect a simple row-split: for 5 rows and batch_size 2, we expect 3 batches:
    # [rows 0-1], [rows 2-3], and [row 4].
    assert len(batches) == 3
    assert batches[0]["response_id"].tolist() == [1, 2]
    assert batches[1]["response_id"].tolist() == [3, 4]
    assert batches[2]["response_id"].tolist() == [5]


def test_batch_task_input_df_iterative(monkeypatch):
    """
    Test the branch where the sample batch token count is above allowed_tokens,
    triggering the iterative row-by-row accumulation logic.
    """
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_iterative,
    )
    # Create a DataFrame with 3 rows.
    df = pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})
    batch_size = 2
    allowed_tokens = 50
    # With dummy_token_length_iterative:
    # - The sample batch (first 2 rows) will yield a token count of 100 (forcing the iterative branch).
    # - For each individual row, row.to_json() is short and returns 10 tokens.
    # The logic will accumulate rows until batch_size is reached.

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key=None)

    # With batch_size=2, we expect the first batch to contain rows 0 and 1, and the second batch to contain row 2.
    assert len(batches) == 2
    assert batches[0]["response_id"].tolist() == [1, 2]
    assert batches[1]["response_id"].tolist() == [3]


def test_batch_task_input_df_partitioning(monkeypatch):
    """
    Test that partitioning works: the DataFrame is split into partitions
    based on the given partition_key before batching.
    """
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_token_length_low,
    )
    # Create a DataFrame with a partition column.
    df = pd.DataFrame(
        {
            "response_id": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
            "group": ["A", "A", "B", "B"],
        }
    )
    batch_size = 1
    allowed_tokens = 100  # Low token count branch, so simple splitting.

    batches = batch_task_input_df(df, allowed_tokens, batch_size, partition_key="group")

    # Expect two partitions: group "A" yields two batches (one row each) and group "B" yields two batches.
    assert len(batches) == 4
    # Check that each batch has a single row and that partitioning by 'group' is preserved.
    group_a_ids = [
        batch["response_id"].iloc[0]
        for batch in batches
        if batch["response_id"].iloc[0] in [1, 2]
    ]
    group_b_ids = [
        batch["response_id"].iloc[0]
        for batch in batches
        if batch["response_id"].iloc[0] in [3, 4]
    ]
    assert sorted(group_a_ids) == [1, 2]
    assert sorted(group_b_ids) == [3, 4]


# Define a dummy calculate_string_token_length that returns a fixed value.
def dummy_calculate_string_token_length(input_text: str, model="gpt-4o") -> int:
    # For our tests, we simulate that the prompt template always uses 50 tokens.
    return 50


# Define a dummy batch_task_input_df that returns a predetermined list of DataFrame batches.
def dummy_batch_task_input_df(
    df: pd.DataFrame, allowed_tokens: int, batch_size: int, partition_key: str | None
):
    # For simplicity, ignore the inputs and return two fixed batches.
    batch1 = pd.DataFrame({"response_id": [1, 2], "text": ["a", "b"]})
    batch2 = pd.DataFrame({"response_id": [3], "text": ["c"]})
    return [batch1, batch2]


@pytest.fixture
def dummy_input_data():
    # Provide some dummy input data (won't matter because we override batch_task_input_df).
    return pd.DataFrame({"response_id": [1, 2, 3], "text": ["a", "b", "c"]})


def test_generate_prompts(monkeypatch, dummy_input_data):
    """
    Test generate_prompts when batch_task_input_df returns two predetermined batches.
    """
    # Monkey-patch dependencies:
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_calculate_string_token_length,
    )
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.batch_task_input_df",
        dummy_batch_task_input_df,
    )

    # Create a dummy prompt template.
    prompt_template = DummyPromptTemplate("dummy template")

    # Call generate_prompts with an extra keyword argument.
    prompts = generate_prompts(
        prompt_template,
        dummy_input_data,
        batch_size=50,
        max_prompt_length=50000,
        partition_key=None,
        extra="foo",
    )

    # We expect two BatchPrompt objects since dummy_batch_task_input_df returns two batches.
    assert len(prompts) == 2

    # For batch1, which has 2 rows, the dummy prompt template returns:
    # "Formatted: 2 responses. foo"
    assert prompts[0].prompt_string == "Prompt with 2 responses. foo"
    # And response IDs should be converted to strings.
    assert prompts[0].response_ids == ["1", "2"]

    # For batch2, which has 1 row:
    assert prompts[1].prompt_string == "Prompt with 1 responses. foo"
    assert prompts[1].response_ids == ["3"]


def test_generate_prompts_with_partition(monkeypatch):
    """
    Test generate_prompts when partitioning is applied.
    We'll simulate partitioning by defining a dummy batch_task_input_df that splits the DataFrame
    based on a partition key.
    """

    def dummy_partition_batch_task_input_df(
        df: pd.DataFrame,
        allowed_tokens: int,
        batch_size: int,
        partition_key: str | None,
    ):
        # Partition the DataFrame by the given key and return each partition as one batch.
        if partition_key:
            partitions = [
                group.reset_index(drop=True) for _, group in df.groupby(partition_key)
            ]
            return partitions
        return [df]

    monkeypatch.setattr(
        "themefinder.llm_batch_processor.calculate_string_token_length",
        dummy_calculate_string_token_length,
    )
    monkeypatch.setattr(
        "themefinder.llm_batch_processor.batch_task_input_df",
        dummy_partition_batch_task_input_df,
    )

    prompt_template = DummyPromptTemplate("dummy template")
    # Create input data that includes a partition key 'group'.
    df = pd.DataFrame(
        {
            "response_id": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
            "group": ["A", "A", "B", "B"],
        }
    )

    prompts = generate_prompts(
        prompt_template,
        df,
        batch_size=50,
        max_prompt_length=50000,
        partition_key="group",
        extra="bar",
    )

    # We expect two batches—one for each unique group.
    assert len(prompts) == 2

    # Since groupby order is not guaranteed, collect the response IDs from both prompts.
    response_ids = {tuple(prompt.response_ids) for prompt in prompts}
    expected = {("1", "2"), ("3", "4")}
    assert response_ids == expected

    # Also verify that the prompt string includes the correct number of responses.
    for prompt in prompts:
        if len(prompt.response_ids) == 2:
            assert prompt.prompt_string == "Prompt with 2 responses. bar"
