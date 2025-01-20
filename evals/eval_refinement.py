import asyncio
from pathlib import Path

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

from themefinder import theme_refinement
from utils import read_and_render


def load_condensed_themes() -> pd.DataFrame:
    parent_dir = Path(__file__).parent
    return pd.read_csv(f"{parent_dir}/data/refinement/eval_condensed_themes.csv")


async def evaluate_refinement():
    dotenv.load_dotenv()
    llm = AzureChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    condensed_themes = load_condensed_themes()
    refined_themes = await theme_refinement(
        condensed_themes,
        llm=llm,
        expanded_question="",
    )
    condensed_themes = condensed_themes[["topic_label", "topic_description"]].to_dict(
        orient="records"
    )
    eval_prompt = read_and_render(
        "refinement_eval.txt",
        {"original_topics": condensed_themes, "neutral_topics": refined_themes},
    )
    response = llm.invoke(eval_prompt)
    print(f"Theme Refinement Eval Results: \n {response.content}")


if __name__ == "__main__":
    asyncio.run(evaluate_refinement())
