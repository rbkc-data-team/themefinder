import ast
import asyncio
import io
import json
import os

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

from metrics import calculate_mapping_metrics
from themefinder import theme_mapping
from utils import download_file_from_bucket


def load_mapped_responses(question: int = 1) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    dotenv.load_dotenv()
    bucket_name = os.getenv("THEMEFINDER_S3_BUCKET_NAME")
    expanded_question = download_file_from_bucket(
        f"app_data/evals/theme_mapping/question_{question}_expanded_question.txt",
        bucket_name=bucket_name,
    ).decode()
    topics = json.loads(
        download_file_from_bucket(
            f"app_data/evals/theme_mapping/question_{question}_topics.json",
            bucket_name=bucket_name,
        )
    )
    cleaned_topics = {}
    for t in topics:
        cleaned_topics[t] = topics[t]["topic_name"] + ": " + topics[t]["rationale"]
    cleaned_topics = pd.DataFrame.from_dict(data=cleaned_topics, orient="index").T
    responses = pd.read_csv(
        io.BytesIO(
            download_file_from_bucket(
                f"app_data/evals/theme_mapping/question_{question}_responses.csv",
                bucket_name=bucket_name,
            )
        )
    )
    responses["topics"] = responses["topics"].apply(ast.literal_eval)
    return expanded_question, cleaned_topics, responses


async def evaluate_mapping(question_num: int | None = None):
    dotenv.load_dotenv()
    llm = AzureChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    questions_to_process = [question_num] if question_num is not None else range(1, 4)
    for i in questions_to_process:
        expanded_question, topics, responses = load_mapped_responses(i)
        result = await theme_mapping(
            responses_df=responses[["response_id", "response"]],
            llm=llm,
            expanded_question=expanded_question,
            refined_themes_df=topics,
        )
        responses = responses.merge(
            result[["response_id", "labels"]], "inner", on="response_id"
        )
        mapping_metrics = calculate_mapping_metrics(
            df=responses, column_one="topics", column_two="labels"
        )
        print(f"Theme Mapping Question {i} Eval Results: \n {mapping_metrics}")


if __name__ == "__main__":
    asyncio.run(evaluate_mapping())
