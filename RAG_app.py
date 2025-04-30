import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keyring
import os 
import snowflake.connector as sf_connector # ( https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect)
from snowflake.connector.pandas_tools import write_pandas # (https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#write_pandas)
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
from shapely.geometry import box
from shapely.ops import unary_union
from PIL import Image, ImageDraw
import cv2
from openai import OpenAI
import base64
from datetime import datetime


def log(message: str) -> None:
    """
    Logs a message to the console. Wrapper function made for easy modification in the future.
    """
    if VERBOSE > 0:
        cur_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f" {cur_datetime}  [LOG]  {message}")

def log2(message: str) -> None:
    """
    Logs a message to the console. Wrapper function made for easy modification in the future.
    """
    if VERBOSE == 2:
        cur_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f" {cur_datetime}  [LOG]  {message}")



def convert_relevant_path_to_absolute_path(relative_path: str) -> str:
    """
    Converts a relative path to an absolute path.
    """
    current_directory = os.getcwd()
    absolute_path = os.path.join(current_directory, relative_path)
    
    return absolute_path


def get_openai_api_key() -> str:
    """
    Retrieves the OpenAI API key from the Windows Credential Manager using the keyring library.
    The credentials are stored under the following keys:
    - openai api: api_key
    """

    try:
        open_ai_api_key = keyring.get_password("openai api", "api_key")
        if open_ai_api_key is None:
            raise ValueError("API key not found in keyring.")
    except:
        print("Please set your OpenAI API key in microsoft generic credential manager with the hostname 'openai api' and username 'api_key'")
        open_ai_api_key = None
    return open_ai_api_key


def get_snowflake_connection() -> sf_connector.cursor:
    """
    Establishes a connection to Snowflake from the Windows Credential Manager using the keyring library.
    The credentials are stored under the following keys:
    - NC_Snowflake_Trial_Account_Name
    - NC_Snowflake_Trial_User_Name
    - NC_Snowflake_Trial_User_Password

    The function returns a Snowflake cursor object for executing SQL queries.
    """
    account_identifier = keyring.get_password('NC_Snowflake_Trial_Account_Name', 'account_identifier')
    user_name = keyring.get_password('NC_Snowflake_Trial_User_Name', "user_name")
    password  = keyring.get_password('NC_Snowflake_Trial_User_Password', user_name)
    database  = "WASHING_MACHINE_MANUALS"
    schema = "PUBLIC"

    log("Connecting to Snowflake with")
    log(f"Account: {account_identifier}")
    log(f"User: {user_name}")
    log(f"Database: {database}")
    log(f"Schema: {schema}")
    log("")
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


def generate_promt_for_openai_api(instructions, input_text, open_ai_client) -> str:
    response = open_ai_client.responses.create(
        model="gpt-4o",
        instructions= instructions,
        input=input_text
    )

    return response


def extract_json_from_llm_output(llm_output_text: str) -> dict:
    try:
        # Look for a code block marked with ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON code block found in the text.")

        raw_json = match.group(1)

        # Optional: Remove trailing commas which are invalid in JSON
        cleaned_json = re.sub(r",\s*([\]}])", r"\1", raw_json)

        parsed = json.loads(cleaned_json)
        return parsed

    except Exception as e:
        print("Failed to extract JSON:", e)
        return {}


def vector_embedding_cosine_similarity_search(input_text: str, cursor: sf_connector.cursor, chunk_size: str = "small") -> pd.DataFrame:
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
            PAGE_END_NUMBER,
            chunk_text,
            VECTOR_COSINE_SIMILARITY({table_name}.EMBEDDING, input.VECTOR) AS COSINE_SIMILARITY
        FROM {table_name}, input
        ORDER BY COSINE_SIMILARITY DESC
        LIMIT 100
    """

    # Important: pass input_text as a parameter, NOT interpolated directly
    cursor.execute(sql, (input_text,))
    return_df = cursor.fetch_pandas_all()

    return return_df


def vector_embedding_cosine_similarity_between_texts(text1: str, text2: str, cursor: sf_connector.cursor) -> float:
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



def find_document_by_machine_name(cursor : sf_connector.cursor, machine_name: str) -> dict:
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


def narrow_down_relevant_chunks(task_chunk_df: pd.DataFrame, document_info: dict) -> pd.DataFrame:
    filtered_task_chunk_df = task_chunk_df[task_chunk_df['DOCUMENT_ID'] == document_info['DOCUMENT_ID']]
    filtered_task_chunk_df = filtered_task_chunk_df.sort_values(by='CHUNK_ORDER', ascending=True)
    filtered_task_chunk_df = filtered_task_chunk_df.head(10)

    return filtered_task_chunk_df


def create_step_by_step_prompt(relevant_chunks_df: pd.DataFrame, user_task: str) -> str:
    """
    Builds a prompt asking the LLM to create a step-by-step guide based on relevant chunks.
    
    Args:
        relevant_chunks_df (pd.DataFrame): DataFrame of retrieved relevant chunks.
        user_task (str): The original user query (e.g., "How do I clean the filter?")
    
    Returns:
        str: Prompt ready for LLM completion
    """

    reference_text = ""
    for i, row in relevant_chunks_df.iterrows():
        page_info = f"(page {row['PAGE_START_NUMBER']})" if 'PAGE_START_NUMBER' in row else ""
        reference_text += f"- [Relevance: {row['COSINE_SIMILARITY']}]: {row['CHUNK_TEXT']}  {page_info}\n\n"
        # Section info could also be included in the prompt if needed.

    instructions = f"""
    You are tasked with writing a clear, cohearent step-by-step guide for a user based on the provided reference content and the task.

    The user wants help with the following task:
    "{user_task}"

    Use only the information provided in the reference content below.
    If any step is ambiguous or missing, note that politely rather than guessing.
    """

    reference_text = f"""
    ### Reference Content:
    {reference_text}

    ### Step-by-Step Guide:
    """
    return instructions, reference_text


def call_openai_api_for_image_description(file_path: str, prompt: str, client: OpenAI) -> str:
    """
    Calls OpenAI API to generate a description for the image using the provided context string.

    Args:
        file_uri (str): URI of the image file.
        context_string (str): Context string for the image.

    Returns:
        str: Generated description for the image.
    """

    with open(file_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text":  prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"}
                ],
            }
        ],
    )

    return response.output_text


def populate_image_descriptions(images_df: pd.DataFrame, open_ai_client: OpenAI, cursor: sf_connector.cursor) -> pd.DataFrame:

    # Iterate through each image and generate a description using OpenAI API
        # For each iteration, context of the image is required. I will use all small chunks of the page of the image, and the image itself.
    for idx, row in images_df.iterrows():
        if len(row["DESCRIPTION"]) > 0:
            log(f"Image ID {row['IMAGE_ID']} already has a description. Skipping...")
            continue # Skip if description already exists
        log(f"Generating description for image ID {row['IMAGE_ID']}...")

        file_location = row["IMAGE_PATH"]
        page_number = row["PAGE"]
        section_id = row["SECTION_ID"]
        document_id = row["DOCUMENT_ID"]
        image_id = row["IMAGE_ID"]

        sql = f"""
        SELECT * 
        FROM CHUNKS_SMALL 
        WHERE PAGE_START_NUMBER = %s AND DOCUMENT_ID = %s
        """

        # Important: pass input_text as a parameter, NOT interpolated directly
        cursor.execute(sql, (page_number,document_id,))
        local_small_chunks = cursor.fetch_pandas_all()

        # Create a context string from the relevant chunks
        context_string = "\n".join(local_small_chunks["CHUNK_TEXT"].tolist())
        prompt = f"""
            This image was extracted from the same page as the context string which is concatenated at the end of this string. 
            Please describe the image in detail, including any relevant information that can be inferred from the context.

            CONTEXT:
            {context_string}
            """

        # Call OpenAI API to generate a description for the image
        description_response = call_openai_api_for_image_description(file_location, prompt, open_ai_client)
        log2(f"Generated description for image ID:{image_id} {file_location}: {description_response}")

        # Store the generated description in the DataFrame
        images_df.at[idx, "DESCRIPTION"] = description_response
        log(f"Updated IMAGE table for image ID:{image_id} with new description")

        # Update the database with the new description
        update_sql = f"""
        UPDATE IMAGES
        SET DESCRIPTION = %s
        WHERE IMAGE_ID = %s
        """
        cursor.execute(update_sql, (description_response, image_id))
        cursor.connection.commit()

    return images_df


def pick_image_based_of_descriptions(image_candidates: pd.DataFrame, step_text: str, open_ai_client: OpenAI) -> str:
    image_options_text = ""
    for _, image_row in image_candidates.iterrows():
        image_id = image_row["IMAGE_ID"]
        image_path = image_row["IMAGE_PATH"]
        description = image_row["DESCRIPTION"]
        image_options_text += f"- Image ID: {image_id}, Path: {image_path}, Description: {description}\n"

    instructions = f"""
    You are tasked with modifying the task in a step by step guide. You will append the most relevant image reference to the step,
    by selecting the most relevant image for the following step in a guide:
    "{step_text}"
    """

    reference_text = f"""
    ### Image Options:
    {image_options_text}
    """

    response = generate_promt_for_openai_api(instructions, input_text, open_ai_client)
    return response.output_text


def create_image_string_descriptors(image_candidates: pd.DataFrame) -> str:
    """
    Creates a string descriptor for each image candidate.

    Args:
        image_candidates (pd.DataFrame): DataFrame containing image candidates.

    Returns:
        list: List of string descriptors for each image candidate.
    """
    image_descriptors = "### Below are the image candidates:\n\n"
    for _, row in image_candidates.iterrows():
        image_id = row["IMAGE_ID"]
        image_path = row["IMAGE_PATH"]
        description = row["DESCRIPTION"]
        page_number = row["PAGE"]
        # image_position = f"X1: {row["IMAGE_X1"]} Y1: {row["IMAGE_Y1"]} X2: {row["IMAGE_X2"]} Y2: {row["IMAGE_Y2"]}"

        image_descriptors += f"IMAGE_ID: {image_id}, PATH: {image_path}, \n Description:\n {description} \n"
    
    return image_descriptors



def add_image_references_to_guide(guide_text: str, filtered_task_chunk_df: pd.DataFrame, open_ai_client: OpenAI, cursor: sf_connector.cursor) -> str:
    """
    Inserts image references into a step-by-step guide based on LLM-evaluated image descriptions.

    Args:
        guide_text (str): Step-by-step markdown text.
        filtered_task_chunk_df (pd.DataFrame): Chunks used to build the guide.
        open_ai_client (OpenAI): Authenticated OpenAI client.
        cursor: Snowflake cursor.

    Returns:
        str: Guide text with images inserted into appropriate steps.
    """
    
    # Populate all images with descriptions on the pages where the relevant chunks are located.
    relevant_pages = filtered_task_chunk_df["PAGE_START_NUMBER"].unique()
    document_id = int(filtered_task_chunk_df["DOCUMENT_ID"].iloc[0]) # Assumes that only 1 document is relevant for the task.

    sql = f"""
        SELECT * 
        FROM IMAGES 
        WHERE PAGE IN ({','.join(map(str, relevant_pages))})
        AND DOCUMENT_ID = %s
    """
    cursor.execute(sql, (document_id,))
    images_df = cursor.fetch_pandas_all()

    log("Populating image descriptions if they don't exist...")
    images_df = populate_image_descriptions(images_df, open_ai_client, cursor)

    image_descriptors = create_image_string_descriptors(images_df)
    user_query = f"{guide_text} \n \n {image_descriptors}"
    log("Calling OpenAI API to add image references to the guide...")

    response = generate_promt_for_openai_api(
        instructions="""
        You are are tasked to modify the step by step guide below, and include the most relevant images.
        You should only include images IFF they are relevant to the step.
        The way you will do this is by adding the IMAGE_ID and PATH to the step.
        The input_text will include a long description of each candidate image, and the step by step guide.
        """, 
        input_text = user_query, 
        open_ai_client = open_ai_client
        )

    return response.output_text



def main_RAG_pipeline(user_query: str, machine_name: str = "N/A" , verbose:int = 1) -> str:
    global VERBOSE
    VERBOSE = 1

    if verbose:
        log("Verbose mode is ON.")
    else:
        log("Verbose mode is OFF.")
        VERBOSE = 0

    # Establishing connections to OpenAI and Snowflake (Windows Credential Manager is used to store credentials)
    cursor = get_snowflake_connection()
    open_ai_api_key = get_openai_api_key() 
    client = OpenAI(api_key = open_ai_api_key)    

    log("Starting RAG pipeline...")
    log(f"User query: {user_query}")
    log("Calling for Response 1: Extracting machine name and task...")
    response_1 = generate_promt_for_openai_api(
        instructions=f"""
        Extract from the following user query:
        1. The machine name or type. Let the key be "machine_name". If the user defined machine name is not "N/A", use that.
        2. A one-sentence description of the task. Let the key be "task".

        User defined machine name: {machine_name}

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

    # Filtering the task_chunk_df to only include chunks related to the found document
    log(f"Pre - Filtered task chunk dataframe: {len(task_chunk_df)}")
    filtered_task_chunk_df = narrow_down_relevant_chunks(task_chunk_df, document_info)
    log(f"Post - Filtered task chunk dataframe: {len(filtered_task_chunk_df)}")

    # Retrieve a step by step response from the LLM using the relevant chunks
    instructions_3, reference_text_3 = create_step_by_step_prompt(filtered_task_chunk_df, task)

    log("Calling for Response 3: Constructing a step by step guide using the relevant chunks...")
    log2("\nReference text:")
    log2(reference_text_3)
    log2("\nInstructions:")
    log2(instructions_3)
    log("\nCalling OpenAI API for Response 3...")
    response_3 = generate_promt_for_openai_api(
        instructions=instructions_3, 
        input_text = reference_text_3, 
        open_ai_client = client
        ).output_text

    log2("Response 3:")
    log2(response_3)

    log("Calling for Response 4: Adding image references to the guide...")
    response_4 = add_image_references_to_guide(response_3, filtered_task_chunk_df, client, cursor)
    log("Response 4:")
    log(response_4)

    return response_4


if __name__ == "__main__":

    
    machine_name = "WGA1420SIN"
    user_query = f"There is often detergent residues on the laundry when i do a fine wash cycle. My washing machine model is {machine_name}. How can I fix this?" 

    main_RAG_pipeline(user_query, machine_name)

