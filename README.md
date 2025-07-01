# ThemeFinder: Topic Modelling Tool for Public Consultations

ThemeFinder is an interactive Streamlit application designed to assist analysts, researchers, and policymakers in exploring and identifying themes or topics within free-text responses collected from public consultations. The tool leverages the `themefinder` Python package and generative AI models to provide a semi-automated theming assistant, making it easier to extract meaningful insights from textual data.

## About ThemeFinder and i.ai

ThemeFinder has been developed by [i.ai](https://ai.gov.uk), an organisation focused on delivering AI-driven solutions for public sector challenges. ThemeFinder is currently under active development and forms a standalone component within the broader *Consult* product. Despite being in development, ThemeFinder offers a powerful toolset for thematic analysis and topic modeling of public consultation data.

## Features

- Upload datasets containing free-text consultation responses.
- Automatically generate thematic topics using advanced topic modeling techniques.
- Interactive interface to review, explore, and export identified themes.
- Designed to partially automate the labor-intensive task of manual theming.
- Probabilistic AI-driven results that can be refined through repeated runs.

## Installation

1. Clone this repository or download the source code.

2. Install the required Python packages. You can use the following command to install the main dependencies:

    ```bash
    pip install streamlit themefinder langchain_openai azure-identity httpx python-dotenv numpy pandas
    ```

3. Ensure you have access to the necessary Azure OpenAI credentials as the app integrates with Azure's generative AI services.

4. Create a `.env` file in the root directory with your Azure OpenAI credentials and any other environment variables required by the app.

## Usage

To run the app locally:

```bash
streamlit run app.py
