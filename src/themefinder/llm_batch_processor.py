import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from tenacity import before, retry, stop_after_attempt, wait_random_exponential

from .themefinder_logging import logger


@dataclass
class BatchPrompt:
    prompt_string: str
    response_ids: list[str]


async def batch_and_run(
    responses_df: pd.DataFrame,
    prompt_template: str | Path | PromptTemplate,
    llm: Runnable,
    batch_size: int = 10,
    partition_key: str | None = None,
    response_id_integrity_check: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Process a DataFrame of responses in batches using an LLM.

    Args:
        responses_df (pd.DataFrame): DataFrame containing responses to be processed.
            Must include a 'response_id' column.
        prompt_template (Union[str, Path, PromptTemplate]): Template for LLM prompts.
            Can be a string (file path), Path object, or PromptTemplate.
        llm (Runnable): LangChain Runnable instance that will process the prompts.
        batch_size (int, optional): Number of responses to process in each batch.
            Defaults to 10.
        partition_key (str | None, optional): Optional column name to group responses
            before batching. Defaults to None.
        response_id_integrity_check (bool, optional): If True, verifies that all input
            response IDs are present in LLM output and retries failed responses individually.
            If False, no integrity checking or retrying occurs. Defaults to False.
        **kwargs (Any): Additional keyword arguments to pass to the prompt template.

    Returns:
        pd.DataFrame: DataFrame containing the original responses merged with the
            LLM-processed results.
    """
    logger.info(f"Running batch and run with batch size {batch_size}")
    prompt_template = convert_to_prompt_template(prompt_template)
    batched_response_dfs = batch_responses(
        responses_df, batch_size=batch_size, partition_key=partition_key
    )
    batch_prompts = generate_prompts(batched_response_dfs, prompt_template, **kwargs)
    llm_responses, failed_ids = await call_llm(
        batch_prompts=batch_prompts,
        llm=llm,
        response_id_integrity_check=response_id_integrity_check,
    )
    processed_responses = process_llm_responses(llm_responses, responses_df)
    if failed_ids:
        new_df = responses_df[responses_df["response_id"].astype(str).isin(failed_ids)]
        processed_failed_responses = await batch_and_run(
            responses_df=new_df,
            prompt_template=prompt_template,
            llm=llm,
            batch_size=1,
            partition_key=partition_key,
            **kwargs,
        )
        return pd.concat(objs=[processed_failed_responses, processed_responses])
    return processed_responses


def load_prompt_from_file(file_path: str | Path) -> str:
    """Load a prompt template from a text file in the prompts directory.

    Args:
        file_path (str | Path): Name of the prompt file (without .txt extension)
            or Path object pointing to the file.

    Returns:
        str: Content of the prompt template file.
    """
    parent_dir = Path(__file__).parent
    with Path.open(parent_dir / "prompts" / f"{file_path}.txt") as file:
        return file.read()


def convert_to_prompt_template(prompt_template: str | Path | PromptTemplate):
    """Convert various input types to a LangChain PromptTemplate.

    Args:
        prompt_template (str | Path | PromptTemplate): Input template that can be either:
            - str: Name of a prompt file in the prompts directory (without .txt extension)
            - Path: Path object pointing to a prompt file
            - PromptTemplate: Already initialized LangChain PromptTemplate

    Returns:
        PromptTemplate: Initialized LangChain PromptTemplate object.

    Raises:
        TypeError: If prompt_template is not one of the expected types.
        FileNotFoundError: If using str/Path input and the prompt file doesn't exist.
    """
    if isinstance(prompt_template, str | Path):
        prompt_content = load_prompt_from_file(prompt_template)
        template = PromptTemplate.from_template(template=prompt_content)
    elif isinstance(prompt_template, PromptTemplate):
        template = prompt_template
    else:
        msg = "Invalid prompt_template type. Expected str, Path, or PromptTemplate."
        raise TypeError(msg)
    return template


def batch_responses(
    responses_df: pd.DataFrame, batch_size: int = 10, partition_key: str | None = None
) -> list[pd.DataFrame]:
    """Split a DataFrame into batches, optionally partitioned by a key column.

    Args:
        responses_df (pd.DataFrame): Input DataFrame to be split into batches.
        batch_size (int, optional): Maximum number of rows in each batch. Defaults to 10.
        partition_key (str | None, optional): Column name to group by before batching.
            If provided, ensures rows with the same partition key value stay together
            and each group is batched separately. Defaults to None.

    Returns:
        list[pd.DataFrame]: List of DataFrame batches, where each batch contains
            at most batch_size rows. If partition_key is used, rows within each
            partition are kept together and batched separately.
    """
    if partition_key:
        grouped = responses_df.groupby(partition_key)
        batches = []
        for _, group in grouped:
            group_batches = [
                group.iloc[i : i + batch_size].reset_index(drop=True)
                for i in range(0, len(group), batch_size)
            ]
            batches.extend(group_batches)
        return batches

    return [
        responses_df.iloc[i : i + batch_size].reset_index(drop=True)
        for i in range(0, len(responses_df), batch_size)
    ]


def generate_prompts(
    response_dfs: list[pd.DataFrame], prompt_template: PromptTemplate, **kwargs: Any
) -> list[BatchPrompt]:
    """Generate a list of BatchPrompts from DataFrames using a prompt template.

    Args:
        response_dfs (list[pd.DataFrame]): List of DataFrames, each containing a batch
            of responses to be processed. Each DataFrame must include a 'response_id' column.
        prompt_template (PromptTemplate): LangChain PromptTemplate object used to format
            the prompts for each batch.
        **kwargs (Any): Additional keyword arguments to pass to the prompt template's
            format method.

    Returns:
        list[BatchPrompt]: List of BatchPrompt objects, each containing:
            - prompt_string: Formatted prompt text for the batch
            - response_ids: List of response IDs included in the batch

    Note:
        The function converts each DataFrame to a list of dictionaries and passes it
        to the prompt template as the 'responses' variable.
    """
    batched_prompts = []

    for df in response_dfs:
        prompt = prompt_template.format(
            responses=df.to_dict(orient="records"), **kwargs
        )
        response_ids = df["response_id"].astype(str).to_list()
        batched_prompts.append(
            BatchPrompt(prompt_string=prompt, response_ids=response_ids)
        )

    return batched_prompts


async def call_llm(
    batch_prompts: list[BatchPrompt],
    llm: Runnable,
    concurrency: int = 10,
    response_id_integrity_check: bool = False,
):
    """Process multiple batches of prompts concurrently through an LLM with retry logic.

    Args:
        batch_prompts (list[BatchPrompt]): List of BatchPrompt objects, each containing a
            prompt string and associated response IDs to be processed.
        llm (Runnable): LangChain Runnable instance that will process the prompts.
        concurrency (int, optional): Maximum number of simultaneous LLM calls allowed.
            Defaults to 10.
        response_id_integrity_check (bool, optional): If True, verifies that all input
            response IDs are present in the LLM output. Failed batches are discarded and
            their IDs are returned for retry. Defaults to False.

    Returns:
        tuple[list[dict[str, Any]], set[str]]: A tuple containing:
            - list of successful LLM responses as dictionaries
            - set of failed response IDs (empty if no failures or integrity check is False)

    Notes:
        - Uses exponential backoff retry strategy with up to 6 attempts per batch
        - Failed batches (when integrity check fails) return None and are filtered out
        - Concurrency is managed via asyncio.Semaphore to prevent overwhelming the LLM
    """
    semaphore = asyncio.Semaphore(concurrency)
    failed_ids: set = set()

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before=before.before_log(logger=logger, log_level=logging.DEBUG),
        reraise=True,
    )
    async def async_llm_call(batch_prompt):
        async with semaphore:
            response = await llm.ainvoke(batch_prompt.prompt_string)
            parsed_response = json.loads(response.content)

            if response_id_integrity_check and not check_response_integrity(
                batch_prompt.response_ids, parsed_response
            ):
                # discard this response but keep track of failed response ids
                failed_ids.update(batch_prompt.response_ids)
                return None

            return parsed_response

    results = await asyncio.gather(
        *[async_llm_call(batch_prompt) for batch_prompt in batch_prompts]
    )
    successful_responses = [
        r for r in results if r is not None
    ]  # ignore discarded responses
    return (successful_responses, failed_ids)


def check_response_integrity(
    input_response_ids: set[str], parsed_response: dict
) -> bool:
    """Verify that all input response IDs are present in the LLM's parsed response.

    Args:
        input_response_ids (set[str]): Set of response IDs that were included in the
            original prompt sent to the LLM.
        parsed_response (dict): Parsed response from the LLM containing a 'responses' key
            with a list of dictionaries, each containing a 'response_id' field.

    Returns:
        bool: True if all input response IDs are present in the parsed response and
            no additional IDs are present, False otherwise.
    """
    response_ids_set = set(input_response_ids)

    returned_ids_set = {
        str(
            element["response_id"]
        )  # treat ids as strings to match response_ids_in_each_prompt
        for element in parsed_response["responses"]
        if element.get("response_id", False)
    }
    # assumes: all input ids ought to be present in output
    if returned_ids_set != response_ids_set:
        logger.info("Failed integrity check")
        logger.info(
            f"Present in original but not returned from LLM: {response_ids_set - returned_ids_set}. Returned in LLM but not present in original: {returned_ids_set -response_ids_set}"
        )
        return False
    return True


def process_llm_responses(
    llm_responses: list[dict[str, Any]], responses: pd.DataFrame
) -> pd.DataFrame:
    """Process and merge LLM responses with the original DataFrame.

    Args:
        llm_responses (list[dict[str, Any]]): List of LLM response dictionaries, where each
            dictionary contains a 'responses' key with a list of individual response objects.
        responses (pd.DataFrame): Original DataFrame containing the input responses, must
            include a 'response_id' column.

    Returns:
        pd.DataFrame: A merged DataFrame containing:
            - If response_id exists in LLM output: Original responses joined with LLM results
              on response_id (inner join)
            - If no response_id in LLM output: DataFrame containing only the LLM results
    """
    responses.loc[:, "response_id"] = responses["response_id"].astype(int)
    unpacked_responses = [
        response
        for batch_response in llm_responses
        for response in batch_response.get("responses", [])
    ]
    task_responses = pd.DataFrame(unpacked_responses)
    if "response_id" in task_responses.columns:
        task_responses["response_id"] = task_responses["response_id"].astype(int)
        return responses.merge(task_responses, how="inner", on="response_id")
    return task_responses
