import streamlit as st
import pandas as pd
import asyncio
import os
import io
import logging
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from themefinder import find_themes
import httpx
from dotenv import load_dotenv
import numpy as np  

st.set_page_config(page_title="ThemeFinder Tool", layout="wide")

st.title("ThemeFinder: Topic Modelling Tool for Public Consultations")
st.markdown(  
    "<p style='font-size:16px; color:black;'>"
    "This is a tool designed as a helpful, first-draft theming assistant."  
    "  Use this tool to partially-automate the theming or topic modelling for free-text fields from a public consultation.  " 
    "  This tool uses generative AI which is probabilistic, meaning that if you were to run this tool multiple times with the same dataset you will likely get different outputs."
    "  If you pass a dataset through the tool and you find there are too many topics created, re-submit the original data and run the tool again.  This tool will generally perform better on the second or third iteration, providing more condensed themes."
    "<br><br>" 
    "  As with any genAI tool, the outputs from this tool are not a final product and must be reviewed by a human.  <b> YOU are respsible for the final product, not the AI tool.</b>"
    "  By using this tool you agree to this and take responsibility for editing and verifying the outputs."
    "<br><br>  This tool sends data to genAI models based in Azure.  Whilst Microsoft has <a href='https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/openai/data-privacy?tabs=azure-portal#:~:text=Azure%20compliance%20offerings.-,Important,-Your%20prompts%20(inputs' target='_blank'>assurances</a> that data is not used to re-train the models, please DO NOT include personally identifiable information in your datasets."   
    "</p>",  
    unsafe_allow_html=True  
)  
  
 
centered_left_aligned_markdown = """
<div style="text-align: left; max-width: 700px; margin: 0 5% 0 15%;  ">
<h3>ThemeFinder pipeline</h3>

<p>ThemeFinder's pipeline consists of five distinct stages, each utilizing a specialized LLM prompt:</p>

<ul style="list-style-type:none; padding-left:0;">
<li><b>Sentiment analysis</b><br>
Analyses the emotional tone and position of each response using sentiment-focused prompts<br>
Provides structured sentiment categorisation based on LLM analysis</li><br>

<li><b>Theme generation</b><br>
Uses exploratory prompts to identify initial themes from response batches<br>
Groups related responses for better context through guided theme extraction</li><br>

<li><b>Theme condensation</b><br>
Employs comparative prompts to combine similar or overlapping themes<br>
Reduces redundancy in identified topics through systematic theme evaluation</li><br>

<li><b>Theme refinement</b><br>
Leverages standardisation prompts to normalise theme descriptions<br>
Creates clear, consistent theme definitions through structured refinement</li><br>

<li><b>Theme target alignment</b><br>
Optional step to consolidate themes down to a target number</li><br>

<li><b>Theme mapping</b><br>
Utilizes classification prompts to map individual responses to refined themes<br>
Supports multiple theme assignments per response through detailed analysis</li>
</ul>

<p>For more detail - see the docs: <a href="https://i-dot-ai.github.io/themefinder/" target="_blank">https://i-dot-ai.github.io/themefinder/</a></p>
</div>
"""

st.markdown(centered_left_aligned_markdown, unsafe_allow_html=True)


# Initialize logs list in session state
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# Custom logging handler to append logs to session state
class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        st.session_state["logs"].append(log_entry)

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Remove all handlers associated with the root logger object.
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
streamlit_handler = StreamlitLogHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
streamlit_handler.setFormatter(formatter)
logger.addHandler(streamlit_handler)

# File uploader for .csv and .xlsx
st.markdown("""  
**Input Data Requirements:**  
  
- The data file must have exactly two columns.  
- The columns must be named **`response`** and **`response_id`** (these should be the headers in the first row).  
- The **`response_id`** must be an integer (number) and cannot contain letters or special characters or the tool will fail to execute the theming.  
""")  
uploaded_file = st.file_uploader("Upload your data file (.csv or .xlsx)", type=["csv", "xlsx"],
                                 help="Upload one question at a time. The document can only have two columns and they MUST be labelled as 'response' for the text based responses and 'response_id' for the unique identifyer.  The unique ID must be a number (integer) and not contain any letters or special characters.",
                                 label_visibility='visible')

# Inputs
n_themes = st.number_input("Enter the target number of themes for this question",
                           help = "This is a target theme/topic number and is input to help guide the model.  It does guarantee that the final output will exactly match this figure (it may be smaller), but instead provides a maximum figure that the topics will be condensed down to.",
                           label_visibility='visible',
                           step=1,
                           format="%d",
                           value=10)

question = st.text_input("Enter your question", 
                         help="This is the survey or consultation question that relates to the responses uploaded above for theming",
                         label_visibility='visible')

system_prompt = st.text_area("Enter system prompt (e.g. directions for theme finding)", 
                             placeholder="You are an AI evaluation tool analyzing survey responses from a local government regarding the rollout of a grant for residents living near the Grenfell Tower disaster.  You look at the responses to the given survey and group the responses into the number of topics stipulated.",
                             help="The system prompt is used as high level instructions for the LLM.  Use this to instruct the tool on specific information relating to the themes/topics you want as an output.  If you do not like the outputs from a theming excersise, try to be more specific in the system prompt.",
                             label_visibility='visible')

custom_categories_input = st.text_area(  
    "Custom Categories (one per line, optional)",  
    help="Enter custom categories to guide theme generation. Each category should be on a new line.",  
    label_visibility='visible'  
)  
  
# Parse input into list of strings  
custom_categories = [cat.strip() for cat in custom_categories_input.split('\n') if cat.strip()]

load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_GPT_MODEL")
model_version = os.getenv("AZURE_MODEL_VERSION")
api_version = os.getenv("OPENAI_API_VERSION")

process_button = st.button("Find Themes")
st.markdown(  
    "<p style='font-size:15px; color:gray;'>"  
    "This can take a few minutes depending on the size of the dataset. A table will appear below when the process is finished."    
    "</p>",  
    unsafe_allow_html=True  
)  

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "theme_df" not in st.session_state:
    st.session_state["theme_df"] = None

def export_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def export_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Themes')
    processed_data = output.getvalue()
    return processed_data

async def run_themefinder(df, question, system_prompt, n_themes, custom_categories=None):
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    llm = AzureChatOpenAI(
        model=model,
        azure_deployment=deployment,
        model_version=model_version,
        azure_endpoint=endpoint,
        temperature=0.1,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        http_client=httpx.Client(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
        openai_api_key=api_key,
    )
    logger.info("Starting theme finding process...")
    result = await find_themes(df, llm, question, system_prompt=system_prompt, target_n_themes=n_themes, custom_categories=custom_categories)
    logger.info("Theme finding process completed.")
    return result
  
# Function to assign random int > 5000 to NaN response_id values  
def fill_na_with_random(df, col="response_id", min_val=5001):  
    na_mask = df[col].isna()  
    n_na = na_mask.sum()  
    if n_na > 0:  
        # Generate random integers > 5000 for each NaN  
        random_ids = np.random.randint(min_val, min_val + 10000, size=n_na)  
        df.loc[na_mask, col] = random_ids  
    df[col] = df[col].astype(int)  
    return df  

def merge_results(result):
    df_sent = pd.DataFrame.from_dict(result['sentiment'])
    df_theme = pd.DataFrame.from_dict(result['themes'])
    df_mapping = pd.DataFrame.from_dict(result['mapping'])
    merged = df_mapping.merge(df_sent[['position', 'response_id']], how='left', on='response_id')

    topic_dict = dict(zip(df_theme['topic_id'], df_theme['topic']))

    def map_labels_to_topics(labels):
        return [topic_dict[label] for label in labels]

    for i in range(len(merged)):
        topics = map_labels_to_topics(merged.at[i, 'labels'])
        for j, topic in enumerate(topics):
            merged.at[i, f'topic_{j+1}'] = topic

    df_exploded = merged.explode('labels')  
    topic_counts = (  
        df_exploded.groupby('labels')['response_id']  
        .nunique()  
        .reset_index()  
        .rename(columns={'labels': 'topic_id', 'response_id': 'response_count'})  
    )  
    df_theme = df_theme.merge(topic_counts, on='topic_id', how='left') 

    return merged, df_theme

if process_button:
    # Clear previous logs on new run
    st.session_state["logs"] = []
    if uploaded_file is None:
        st.error("Please upload a data file first.")
    elif not question:
        st.error("Please enter a question.")
    elif not system_prompt:
        st.error("Please enter a system prompt.")
    else:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                #df["response"] = df["response"].fillna(" ")
                df = fill_na_with_random(df, "response_id")   
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read the file: {e}")
            df = None

        if df is not None:
            with st.spinner("Finding themes..."):
                result = asyncio.run(run_themefinder(df, question, system_prompt, n_themes, custom_categories))
            merged_df, df_theme = merge_results(result)
            st.success("Themes found and merged successfully!")
            st.session_state["results_df"] = merged_df
            st.session_state["theme_df"] = df_theme  
            #st.session_state["theme_df"] = pd.DataFrame.from_dict(result['themes'])

if st.session_state["results_df"] is not None:
    st.download_button(
        label="Download results as CSV",
        data=export_df_to_csv(st.session_state["results_df"]),
        file_name="themefinder_results.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download results as Excel",
        data=export_df_to_excel(st.session_state["results_df"]),
        file_name="themefinder_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if st.session_state.get("theme_df") is not None:
    st.subheader("Discovered Themes")
    st.markdown(  
    "<p style='font-size:15px; color:gray;'>"  
    "The counts included in the source_topic_count from the table below are a sum of all the original topics (theme generation) that have been condensed and refined into the topics shown in the table. See pipeline above for details."
    "  The column called response_count is a sum of all the responses that are attributed to that specific theme/topic.  A comment can be attributed to more than one topic."    
    "</p>",  
    unsafe_allow_html=True  
)  
    theme_df_display = st.session_state["theme_df"].copy()
    for col in theme_df_display.columns:
        if theme_df_display[col].apply(lambda x: isinstance(x, (list, dict))).any():
            theme_df_display[col] = theme_df_display[col].apply(str)
    # Convert to HTML and display
    st.markdown(theme_df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button(  
        label="Download themes as Excel",  
        data=export_df_to_excel(st.session_state["theme_df"]),  
        file_name="themefinder_themes.xlsx",  
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  
    )  


# Display logs in a multiline text area after processing completes
st.subheader("Process Logs")
if st.session_state["logs"]:
    logs_text = "\n".join(st.session_state["logs"])
    st.text_area("Logs", value=logs_text, height=300)
else:
    st.write("No logs available yet.")

