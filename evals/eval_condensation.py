import asyncio
from pathlib import Path

import dotenv
import pandas as pd
from langchain_openai import AzureChatOpenAI

from themefinder import theme_condensation
from utils import read_and_render


def load_generated_themes() -> tuple[pd.DataFrame, str]:
    data_dir = Path(__file__).parent / "data/condensation"
    themes = pd.read_csv(data_dir / "eval_generated_themes.csv")
    with (data_dir / "expanded_question.txt").open() as f:
        expanded_question = f.read()
    return themes, expanded_question


async def evaluate_condensation():
    dotenv.load_dotenv()
    llm = AzureChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    themes, expanded_question = load_generated_themes()
    condensed_themes = await theme_condensation(
        themes,
        llm=llm,
        expanded_question=expanded_question,
    )
    condensed_themes = condensed_themes[["topic_label", "topic_description"]].to_dict(
        orient="records"
    )
    eval_prompt = read_and_render(
        "condensation_eval.txt",
        {"original_topics": themes, "condensed_topics": condensed_themes},
    )
    response = llm.invoke(eval_prompt)
    print(condensed_themes)
    print(f"Theme Condensation Eval Results: \n {response.content}")


if __name__ == "__main__":
    asyncio.run(evaluate_condensation())
