import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keyring
import os 
import snowflake.connector as sf_connector # ( https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect)
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.evaluation import load_evaluator
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import time
import re
import json

from io import BytesIO
import fitz 
from PIL import Image, ImageDraw
import cv2
from openai import OpenAI
from datetime import datetime


def log(message):
    """
    Logs a message to the console. Wrapper function made for easy modification in the future.
    """
    cur_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" {cur_datetime}  [LOG]  {message}")


def get_openai_api_key():
    try:
        open_ai_api_key = keyring.get_password("openai api", "api_key")
        if open_ai_api_key is None:
            raise ValueError("API key not found in keyring.")
    except:
        print("Please set your OpenAI API key in microsoft generic credential manager with the hostname 'openai api' and username 'api_key'")
        open_ai_api_key = None
    return open_ai_api_key


def get_snowflake_connection(account_identifier, user_name, password, database, schema):
    try:
        conn = sf_connector.connect(
            account=account_identifier,
            user=user_name,
            password=password,
            database=database,
            schema=schema
        )
        
        cursor = conn.cursor()
        cursor.execute(f" USE DATABASE {database}; ")
        cursor.execute(f" USE SCHEMA {schema}; ")
        return cursor
    except Exception as e:
        raise(f"Error connecting to Snowflake: {e}")


def generate_promt_for_openai_api(instructions, input_text, open_ai_client):
    response = open_ai_client.responses.create(
        model="gpt-4o",
        instructions= instructions,
        input=input_text
    )

    return response


def extract_json_from_llm_output(llm_output_text):
    """
    Extracts the first JSON object from raw LLM output text and returns it as a dictionary.

    Assumptions:
    - The JSON is flat (non-nested).
    - The JSON block is enclosed by the first '{' and the last '}'.

    Args:
        llm_output_text (str): Raw output text from LLM.

    Returns:
        dict: Parsed dictionary from JSON content.

    Raises:
        ValueError: If JSON cannot be parsed.
    """
    try:
        # Locate the first '{' and the last '}'
        start_idx = llm_output_text.find("{")
        end_idx = llm_output_text.rfind("}") + 1

        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in the output text.")

        json_text = llm_output_text[start_idx:end_idx]

        # Parse the JSON string
        parsed_dict = json.loads(json_text)

        if not isinstance(parsed_dict, dict):
            raise ValueError("Extracted JSON is not a dictionary.")

        return parsed_dict

    except Exception as e:
        raise ValueError(f"Failed to parse JSON from LLM output: {e}")



def vector_embedding_cosine_similarity_search(input_text, cursor, chunk_size: str = "small"):
    """
    chunk_size: str = "small" or "large" - refers to the database table to search in.
    Searches for similar chunks based on cosine similarity.
    Returns a Pandas DataFrame with results.
    """
    if chunk_size == "small":
        table_name = "CHUNKS_SMALL"
    elif chunk_size == "large":
        table_name = "CHUNKS_LARGE"
    else:
        raise ValueError("chunk_size must be 'small' or 'large'.")

    sql = f"""
    WITH input AS (
        SELECT
            SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', %s) AS VECTOR
        )
        SELECT
            document_id,
            chunk_id,
            CHUNK_ORDER,
            PAGE_START_NUMBER,
            PAGE_END_NUMBER
            chunk_text,
            VECTOR_COSINE_SIMILARITY({table_name}.EMBEDDING, input.VECTOR) AS COSINE_SIMILARITY
        FROM {table_name}, input
        ORDER BY COSINE_SIMILARITY DESC
        LIMIT 50
    """

    # Important: pass input_text as a parameter, NOT interpolated directly
    cursor.execute(sql, (input_text,))
    return_df = cursor.fetch_pandas_all()

    return return_df


def vector_embedding_cosine_similarity_between_texts(text1, text2, cursor):
    """
    Computes cosine similarity between two input texts using Snowflake Arctic Embedding.

    Args:
        text1 (str): First input text.
        text2 (str): Second input text.
        cursor: Snowflake database cursor.

    Returns:
        float: Cosine similarity between text1 and text2 (range -1 to 1).
    """
    sql = f"""
    WITH embeddings AS (
    SELECT 
        SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', %s) AS vector1,
        SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', %s) AS vector2
    )
    SELECT VECTOR_COSINE_SIMILARITY(vector1, vector2) AS cosine_similarity
    FROM embeddings
    """

    cursor.execute(sql, (text1, text2))
    result_df = cursor.fetch_pandas_all()

    # Return the single float value
    return result_df['COSINE_SIMILARITY'].iloc[0]



def find_document_by_machine_name(cursor, machine_name):
    """
    Attempts to find a document by machine name using the DOCUMENTS table.
    NOTE: THIS FUNCTION SHOULD BE REPLACED BY A BETTER FUNCTION THAT FIND MACHINE NAMES USING METADATA.
    
    First, tries simple case-insensitive substring matching against DOCUMENT_NAME.
    If no match is found, uses cosine similarity (via embeddings) to choose the best match.
    
    Args:
        cursor: Snowflake DB cursor.
        machine_name (str): The machine name to search for.
        
    Returns:
        dict: A dictionary with keys "DOCUMENT_NAME" and "DOCUMENT_ID" corresponding
              to the best matching document.
    """
    # 1. Retrieve all documents
    cursor.execute("""
        SELECT DOCUMENT_NAME, DOCUMENT_ID 
        FROM DOCUMENTS;
    """)
    documents_df = cursor.fetch_pandas_all()
    
    # 2. Attempt to match using simple string matching (case insensitive)
    for _, row in documents_df.iterrows():
        doc_name = row['DOCUMENT_NAME']
        if machine_name.lower() in doc_name.lower() or doc_name.lower() in machine_name.lower():
            return {"DOCUMENT_NAME": doc_name, "DOCUMENT_ID": row["DOCUMENT_ID"]}
    
    # 3. If no string match was found, use cosine similarity via embeddings.
    best_similarity = -1.0  # cosine similarity ranges from -1 to 1.
    best_match = None
    for _, row in documents_df.iterrows():
        doc_name = row['DOCUMENT_NAME']
        # Use the cosine_similarity_between_texts function to compute similarity.
        similarity = vector_embedding_cosine_similarity_between_texts(machine_name, doc_name, cursor)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = {"DOCUMENT_NAME": doc_name, "DOCUMENT_ID": row["DOCUMENT_ID"]}
    
    return best_match



def main_RAG_pipeline(cursor, open_ai_api_key, database, schema, user_query):
    client = OpenAI(api_key = open_ai_api_key)

    log("Starting RAG pipeline...")
    log(f"User query: {user_query}")
    log("Calling for Response 1: Extracting machine name and task...")
    response_1 = generate_promt_for_openai_api(
        instructions="""
        Extract from the following user query:
        1. The machine name or type. Let the key be "machine_name".
        2. A one-sentence description of the task. Let the key be "task".

        Return as JSON.
        User query: 
        """, 
        input_text = user_query, 
        open_ai_client = client
        )

    response_1 = extract_json_from_llm_output(response_1.output_text)
    machine_name = response_1['machine_name']
    task = response_1['task']
    log(f"Extracted machine name: {machine_name}")
    log(f"Extracted task: {task}")

    log("Finding document ID in snowflake database...")
    document_info = find_document_by_machine_name(cursor, machine_name)

    log(f"Document ID: {document_info['DOCUMENT_ID']}")
    log(f"Document Name: {document_info['DOCUMENT_NAME']}")
    log("Calling for Response 2: Finding most relevant chunks of data to solve the task...")
    task_chunk_df = vector_embedding_cosine_similarity_search(input_text = task, cursor = cursor, chunk_size = "small")

    print(task_chunk_df.head(10))
    print("\n")

    filtered_task_chunk_df = task_chunk_df[task_chunk_df['DOCUMENT_ID'] == document_info['DOCUMENT_ID']]
    print(filtered_task_chunk_df.head(10))
    


if __name__ == "__main__":

    VERBOSE = True
    if VERBOSE:
        log("Verbose mode is ON.")
    machine_name = "WGA1420SIN"
    user_query = f"There is often detergent residues on the laundry when i do a fine wash cycle. My washing machine model is {machine_name}. How can I fix this?" 

    account_identifier = keyring.get_password('NC_Snowflake_Trial_Account_Name', 'account_identifier')
    user_name = "EMHALDEMO1" # Change this to your Snowflake user name
    password = keyring.get_password('NC_Snowflake_Trial_User_Password', user_name)
    database = "WASHING_MACHINE_MANUALS"
    schema = "PUBLIC"

    cursor = get_snowflake_connection(account_identifier, user_name, password, database, schema)

    # Using OpenAI API GPT-4o
    open_ai_api_key = get_openai_api_key() 
    client = OpenAI(api_key = open_ai_api_key)    

    main_RAG_pipeline(cursor, open_ai_api_key, database, schema, user_query)