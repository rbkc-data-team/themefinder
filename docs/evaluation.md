# Evaluation

The `evals/` directory contains our benchmarking evaluation suite used to measure system performance. 

**Note on data access**: At present, these can only be used internally within i.AI. The `eval_mapping` and `eval_sentiment` evaluations utilize sensitive data stored in our AWS environment. These specific evaluations will only function with proper AWS account access and credentials. Similarly, the `make run_evals` command assumes you have AWS access configured.

These evaluations use the Azure Open AI endpoint.

## Running the evaluations

Set your environment variables: copy `.env.example` to `.env` and populate with the name of the S3 bucket, and the details for the Azure endpoint.

Install packages for this repo: `poetry install`.

Ensure you have AWS access set-up, and assume your AWS role to allow you to access the data.

These evaluations can be executed either:
- By running `make run_evals` to execute the complete evaluation suite (or `poetry run make run_evals` if you're using `poetry`)
- By directly running individual evaluation files that begin with `eval_` prefix

Note that the evals specifically use GPT-4o, and JSON structured output.
