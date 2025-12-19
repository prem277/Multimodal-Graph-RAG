# Chat-Your-Data

Create a ChatGPT like experience over your custom docs using [LangChain](https://github.com/langchain-ai/langchain).

## Step 0: Install requirements

`pip install -r requirements.txt`

## Step 1: set your janusgraph

Move to janusgraph\bin floder

Run: gremlin-server.bat

## Step 2: Ingest your data

Run: `python ingest_data.py`

## Query data

Custom prompts are used to ground the answers in the state of the union text file.

## Running the Application

By running `python app.py` from the command line you can easily interact with your ChatGPT over your own data.
