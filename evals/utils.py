import io
from pathlib import Path
from typing import Any

import boto3
from langchain_core.prompts import PromptTemplate


def load_prompt_from_file(file_path: str | Path) -> str:
    """Load a prompt template from a file in the 'prompts' directory.

    Args:
        file_path (str | Path): Path or string pointing to the file name within the prompts directory

    Returns:
        str: The contents of the prompt file
    """
    parent_dir = Path(__file__).parent
    with Path.open(parent_dir / "prompts" / f"{file_path}") as file:
        return file.read()


def read_and_render(prompt_path: str | Path, kwargs: Any = None) -> str:
    """Load a prompt template from a file and optionally render it with provided variables.

    Args:
        prompt_path (str | Path): Path or string pointing to the prompt template file
        kwargs (Any, optional): Dictionary of variables to render in the template

    Returns:
        str: The rendered prompt
    """
    prompt_content = load_prompt_from_file(prompt_path)
    template = PromptTemplate.from_template(template=prompt_content)
    if kwargs:
        return template.format(**kwargs)
    return template.format()


def download_file_from_bucket(
    object_key: str,
    bucket_name: str,
    region_name: str = "eu-west-2",
) -> bytes:
    """Download a file from an S3 bucket to memory using credentials managed by AWS Vault.

    Args:
        object_key (str): Key of the object in the bucket (include prefix if required,
            e.g. "dir1/dir2/object_name")
        bucket_name (str): Name of the S3 bucket (not the ARN)
        region_name (str): AWS region name (optional if using default configuration)

    Returns:
        bytes: content of the file in memory
    """
    s3_client = boto3.client("s3", region_name=region_name)
    file_content = io.BytesIO()
    s3_client.download_fileobj(bucket_name, object_key, file_content)
    return file_content.getvalue()
