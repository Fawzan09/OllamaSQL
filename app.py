from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import requests
import ollama
import duckdb
import json
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from http import HTTPStatus
from time import sleep
import traceback
import re
import os
from werkzeug.utils import secure_filename
import uuid
from threading import Thread
import markdown

# Constants
OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 10
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@dataclass
class AIResponse:
    """Data class to structure AI agent responses"""
    sql_query: Optional[str]
    analysis: str
    sql_results: Optional[Union[List[Dict], Dict]]

class DataFrameManager:
    """Manages DataFrame operations and state"""
    @staticmethod
    def initialize_session_state() -> None:
        # Reset uploaded_files to list format if it's still a dict from old sessions
        if 'uploaded_files' in session and isinstance(session['uploaded_files'], dict):
            session['uploaded_files'] = list(session['uploaded_files'].keys())
        elif 'uploaded_files' not in session:
            session['uploaded_files'] = []
            
        if 'selected_df_file_name' not in session:
            session['selected_df_file_name'] = None
        if 'selected_df_columns' not in session:
            session['selected_df_columns'] = []
        if 'selected_columns' not in session:
            session['selected_columns'] = []
        if 'vector_indices_created' not in session:
            session['vector_indices_created'] = False
        if 'embedding_model_name' not in session:
            session['embedding_model_name'] = None
        if 'similarity_metric' not in session:
            session['similarity_metric'] = 'cosine'
        if 'model_name' not in session:
            session['model_name'] = None
        if 'temperature' not in session:
            session['temperature'] = 0.7

class SQLExecutor:
    """Handles SQL query execution"""
    @staticmethod
    def execute_query(query: str) -> str:
        try:
            selected_file = session.get('selected_df_file_name')
            if not selected_file:
                return json.dumps({"error": "No dataframe selected"})
            
            # Load DataFrame from file
            df = get_dataframe_from_file(selected_file)
            if df is None:
                return json.dumps({"error": "Could not load dataframe"})
            
            with duckdb.connect() as conn:
                conn.register('selected_df', df)
                result_df = conn.execute(query).df()
                return json.dumps(result_df.to_dict(orient='records'))
        except Exception as e:
            return json.dumps({"error": str(e)})

class OllamaService:
    """Handles interactions with Ollama API"""
    @staticmethod
    async def get_ai_response(
        query: str,
        model_name: str,
        embedding_model_name: str,
        temperature: float,
        similarity_metric: str,
        selected_df_dict: dict
    ) -> AIResponse:
        client = ollama.AsyncClient()
        
        df = pd.DataFrame(selected_df_dict)
        num_rows = df.shape[0]
        
        # Build detailed columns info
        columns_info_lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            line = f"- '{col}': data type is {dtype}"
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                line += f", min: {min_val}, max: {max_val}, mean: {mean_val}"
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                unique_vals = df[col].dropna().unique()[:3]
                line += f", sample values: {list(unique_vals)}"
            columns_info_lines.append(line)
        
        columns_info_str = '\n'.join(columns_info_lines)
        
        # Determine the appropriate distance function
        distance_function = {
            'cosine': 'array_cosine_distance',
            'l2sq': 'array_distance'
        }[similarity_metric]

        # Construct the system prompt
        system_prompt = f"""
        You are a data analysis assistant. You have access to a table 'selected_df' with {num_rows} rows. The table has the following columns:

        {columns_info_str}

        Your tasks are:

        1. **Analyze the user's question** and determine whether it requires semantic similarity search (using vector embeddings) or can be answered with a standard SQL query.

        2. **If the question requires semantic similarity search**:
        - Generate an embedding for the user's query using the embedding model.
        - Construct and execute a SQL query that uses the appropriate distance function to perform a vector similarity search.
        - Use '<query_embedding>' as a placeholder in the SQL query for the query embedding.

        3. **If the question can be answered with a standard SQL query**:
        - Generate and execute the SQL query without involving embeddings.

        4. **If the user's question requires both standard SQL operations and semantic similarity search**, construct a combined SQL query that includes both.

        5. Ensure that all generated SQL queries are syntactically correct and consider the SQL execution environment.

        6. Provide the SQL query in a code block labeled as SQL. For example:

        ```sql
        SELECT * FROM selected_df WHERE ...
        ```
        7. Important Instructions:
        You must include the SQL query in your response.
        Enclose the SQL query within triple backticks and label it as 'sql', like so:
        ```sql
        SELECT * FROM ...
        ```
        Do not omit the SQL query or change its format.
        Failure to include the SQL query in the specified format will prevent the application from functioning correctly.
        Always provide a concise summary or answer to the user's question based on the query results.
        8. Always provide a concise summary or answer to the user's question based on the query results.

        Examples:

        If the user's question is "order heat or AC related reviews based on their overall rating (worst first)", the assistant should generate:
        ```sql
        SELECT * FROM selected_df WHERE review_text LIKE '%heat%' OR review_text LIKE '%AC%' ORDER BY overall_rating ASC;
        ```

        If the user's question is "Find products similar to 'wireless earbuds'", the assistant should perform a vector similarity search using the embeddings:
        ```sql
        SELECT * FROM selected_df
        ORDER BY {distance_function}(embedding, ARRAY[<query_embedding>]::FLOAT[])
        LIMIT 5;
        ```

        If an error occurs during execution, analyze the error message and adjust the query accordingly. Always return both the final SQL query and its results in your response.

        """

        # Send the prompt to the model and stream the response
        assistant_response_text = ""
        try:
            # Await the coroutine to get the async generator
            async_gen = await client.generate(
                model=model_name,
                prompt=system_prompt + "\nUser Question:\n" + query,
                options={"temperature": temperature},
                stream=True
            )
            async for part in async_gen:
                chunk = part.get('response', '')
                assistant_response_text += chunk
        except Exception as e:
            print(f"Error during assistant generation: {e}")
            print(traceback.format_exc())
            return AIResponse(sql_query=None, analysis="", sql_results=None)
        
        # Extract the SQL query from the assistant's response
        sql_query = OllamaService.extract_sql_query(assistant_response_text)
        
        if sql_query is None:
            return AIResponse(sql_query=None, analysis=assistant_response_text, sql_results=None)
        
        # Check if the SQL query uses embeddings
        uses_embeddings = 'embedding' in sql_query.lower()
        
        # If embeddings are used, generate query embedding
        if uses_embeddings:
            try:
                embedding_response = await client.embed(
                    model=embedding_model_name,
                    input=[query]
                )
                query_embedding = embedding_response['embedding'][0] if 'embedding' in embedding_response else embedding_response['embeddings'][0]
                query_embedding_str = ','.join(map(str, query_embedding))
                # Replace placeholder in SQL query
                sql_query = sql_query.replace('<query_embedding>', query_embedding_str)
            except Exception as e:
                print(f"Error during query embedding: {e}")
                print(traceback.format_exc())
                return AIResponse(sql_query=sql_query, analysis="", sql_results=None)
        
        # Execute SQL query
        sql_results_json = SQLExecutor.execute_query(sql_query)
        sql_results = json.loads(sql_results_json)

        # Prepare the final analysis
        final_prompt = f"""
Based on the user's question and the query results, please provide a concise summary or answer to the user's question.

SQL Query:
{sql_query}

SQL Results:
{json.dumps(sql_results, indent=2)}
"""

        # Stream the final analysis response
        assistant_final_response = ""
        try:
            # Await the coroutine to get the async generator
            async_gen = await client.generate(
                model=model_name,
                prompt=final_prompt,
                options={"temperature": temperature},
                stream=True
            )
            async for part in async_gen:
                chunk = part.get('response', '')
                assistant_final_response += chunk
        except Exception as e:
            print(f"Error during final response generation: {e}")
            print(traceback.format_exc())
            assistant_final_response = ""

        return AIResponse(
            sql_query=sql_query,
            analysis=assistant_final_response,
            sql_results=sql_results
        )

    @staticmethod
    def extract_sql_query(assistant_response: str) -> Optional[str]:
        # Try to find any SQL code block
        matches = re.findall(r'```sql\s*(.*?)```', assistant_response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        else:
            # Try to find any code block if SQL label is missing
            matches = re.findall(r'```(.*?)```', assistant_response, re.DOTALL | re.IGNORECASE)
            for code in matches:
                # Simple heuristic to check if code looks like SQL
                if any(keyword in code.upper() for keyword in ['SELECT', 'FROM', 'WHERE']):
                    return code.strip()
        return None

    @staticmethod
    def get_embedding(text: str, model_name: str) -> List[float]:
        # Use the Ollama embed method
        embedding_response = ollama.embed(model=model_name, input=[text])
        # embeddings is a list of embeddings, we need the first one
        query_embedding = embedding_response['embedding'][0] if 'embedding' in embedding_response else embedding_response['embeddings'][0]
        return query_embedding

    @staticmethod
    def get_available_models() -> List[str]:
        try:
            response = requests.get(f"{OLLAMA_API_BASE_URL}/api/tags")
            if response.status_code == HTTPStatus.OK:
                return [model['name'] for model in response.json()['models']]
            return []
        except requests.exceptions.RequestException:
            return []

    @staticmethod
    def is_server_running() -> bool:
        try:
            requests.get(
                f"{OLLAMA_API_BASE_URL}/api/tags",
                timeout=OLLAMA_TIMEOUT_SECONDS
            )
            return True
        except requests.exceptions.RequestException:
            return False

# Helper functions for Flask application
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dataframe_from_file(filename):
    """Load DataFrame from uploaded file"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def save_dataframe_to_file(df, filename):
    """Save DataFrame to file and return filename"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df.to_csv(file_path, index=False)
    return filename

def create_vector_indices():
    """Create vector indices for the selected DataFrame"""
    selected_file = session.get('selected_df_file_name')
    if not selected_file:
        return False, "No DataFrame selected"
    
    df = get_dataframe_from_file(selected_file)
    if df is None:
        return False, "Could not load DataFrame"
    
    selected_columns = session.get('selected_columns', [])
    if not selected_columns:
        return False, "No columns selected for embedding"

    embeddings = []
    
    for idx, row in df.iterrows():
        row_texts = [str(row[col]) for col in selected_columns if pd.notnull(row[col])]
        combined_text = ' '.join(row_texts).strip()
        if not combined_text:
            combined_text = " "  # Avoid empty strings
        try:
            # Embed the text individually
            embedding_response = ollama.embed(model=session['embedding_model_name'], input=[combined_text])
            query_embedding = embedding_response['embedding'][0] if 'embedding' in embedding_response else embedding_response['embeddings'][0]
            embedding = [float(value) for value in query_embedding]  # Convert to list of floats
            embeddings.append(embedding)
        except Exception as e:
            return False, f"Error during embedding at index {idx}: {e}"

    # Ensure embeddings length matches DataFrame length
    if len(embeddings) != len(df):
        return False, f"Error: Number of embeddings ({len(embeddings)}) does not match number of DataFrame rows ({len(df)})"

    # Add embeddings to DataFrame
    df = df.copy()
    df['embedding'] = embeddings

    # Determine the embedding dimension
    embedding_length = len(embeddings[0])

    # Create a physical table in DuckDB
    conn = duckdb.connect()
    conn.execute("INSTALL 'vss';")  # Install the VSS extension
    conn.execute("LOAD 'vss';")     # Load the VSS extension

    # Define the schema for the table
    dtype_mapping = {
        'object': 'VARCHAR',
        'int64': 'BIGINT',
        'float64': 'DOUBLE',
        # Add other type mappings as needed
    }

    # Build the column definitions
    column_defs = []
    for col in df.columns:
        if col == 'embedding':
            column_defs.append(f"{col} FLOAT[{embedding_length}]")
        else:
            pandas_dtype = df[col].dtype.name
            duckdb_type = dtype_mapping.get(pandas_dtype, 'VARCHAR')  # Default to VARCHAR if type not found
            column_defs.append(f"{col} {duckdb_type}")

    # Create the table with explicit schema
    conn.execute(f"""
        CREATE TABLE selected_df (
            {', '.join(column_defs)}
        )
    """)

    # Insert data into the table
    data = df.to_records(index=False).tolist()
    conn.executemany(f"INSERT INTO selected_df VALUES ({', '.join(['?' for _ in df.columns])})", data)

    # Create vector index on the base table
    conn.execute(f"""
        CREATE INDEX hnsw_idx ON selected_df USING HNSW (embedding) WITH (metric = '{session['similarity_metric']}')
    """)

    # Save the updated DataFrame with embeddings
    save_dataframe_to_file(df, f"embedded_{selected_file}")
    session['vector_indices_created'] = True
    conn.close()
    
    return True, "Vector indices created successfully"

def run_async_query(query, model_name, embedding_model_name, temperature, similarity_metric, selected_file):
    """Run async query in a separate thread"""
    df = get_dataframe_from_file(selected_file)
    if df is None:
        return AIResponse(sql_query=None, analysis="Error: Could not load DataFrame", sql_results=None)
    
    selected_df_dict = df.to_dict('records')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            OllamaService.get_ai_response(
                query, model_name, embedding_model_name, temperature, similarity_metric, selected_df_dict
            )
        )
        return result
    finally:
        loop.close()

# Flask Routes
@app.route('/')
def index():
    DataFrameManager.initialize_session_state()
    
    # Get available models
    models = []
    if OllamaService.is_server_running():
        models = OllamaService.get_available_models()
    
    # Prepare uploaded files data for display
    uploaded_files_data = {}
    uploaded_files = session.get('uploaded_files', [])
    for filename in uploaded_files:
        df = get_dataframe_from_file(filename)
        if df is not None:
            # Convert to HTML table
            uploaded_files_data[filename] = df.head(100).to_html(classes='table table-striped table-hover', table_id=f'table-{filename}', escape=False)
    
    return render_template('index.html', 
                         models=models,
                         uploaded_files_data=uploaded_files_data)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    session['model_name'] = request.form.get('model_name')
    session['embedding_model_name'] = request.form.get('embedding_model_name')
    session['similarity_metric'] = request.form.get('similarity_metric')
    session['temperature'] = float(request.form.get('temperature', 0.7))
    session.permanent = True
    flash('Settings updated successfully', 'success')
    return redirect(url_for('index'))

@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'csv_files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('csv_files')
    uploaded_count = 0
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                # Read CSV directly from memory
                df = pd.read_csv(file)
                
                # Save to file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                df.to_csv(file_path, index=False)
                
                # Store filename in session
                if 'uploaded_files' not in session:
                    session['uploaded_files'] = []
                if filename not in session['uploaded_files']:
                    session['uploaded_files'].append(filename)
                
                session.permanent = True
                uploaded_count += 1
            except Exception as e:
                flash(f'Error reading {filename}: {str(e)}', 'error')
    
    if uploaded_count > 0:
        flash(f'Successfully uploaded {uploaded_count} file(s)', 'success')
    
    return redirect(url_for('index'))

@app.route('/select_file', methods=['POST'])
def select_file():
    selected_file = request.form.get('selected_file')
    if selected_file and selected_file in session.get('uploaded_files', []):
        session['selected_df_file_name'] = selected_file
        
        # Load DataFrame to get column names
        df = get_dataframe_from_file(selected_file)
        if df is not None:
            session['selected_df_columns'] = df.columns.tolist()
        else:
            session['selected_df_columns'] = []
            
        session['vector_indices_created'] = False
        session['selected_columns'] = []
        session.permanent = True
        flash(f'Selected file: {selected_file}', 'success')
    
    return redirect(url_for('index'))

@app.route('/select_columns', methods=['POST'])
def select_columns():
    selected_columns = request.form.getlist('selected_columns')
    session['selected_columns'] = selected_columns
    session['vector_indices_created'] = False
    session.permanent = True
    flash(f'Selected {len(selected_columns)} column(s) for embedding', 'success')
    return redirect(url_for('index'))

@app.route('/create_indices', methods=['POST'])
def create_indices():
    success, message = create_vector_indices()
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
    return redirect(url_for('index'))

@app.route('/submit_query', methods=['POST'])
def submit_query():
    user_query = request.form.get('user_query', '').strip()
    
    if not user_query:
        flash('Please enter a query', 'error')
        return redirect(url_for('index'))
    
    if not session.get('vector_indices_created'):
        flash('Please create vector indices before querying', 'error')
        return redirect(url_for('index'))
    
    if not OllamaService.is_server_running():
        flash('Ollama server is not running', 'error')
        return redirect(url_for('index'))
    
    try:
        # Run the async query
        response = run_async_query(
            user_query,
            session.get('model_name'),
            session.get('embedding_model_name'),
            session.get('temperature', 0.7),
            session.get('similarity_metric', 'cosine'),
            session.get('selected_df_file_name')
        )
        
        # Prepare query results for display
        query_results_html = None
        if response.sql_results and isinstance(response.sql_results, list):
            results_df = pd.DataFrame(response.sql_results)
            query_results_html = results_df.to_html(classes='table table-striped table-hover', escape=False)
        
        # Convert markdown to HTML for analysis
        if response.analysis:
            response.analysis = markdown.markdown(response.analysis)
        
        # Get uploaded files data for display
        uploaded_files_data = {}
        uploaded_files = session.get('uploaded_files', [])
        for filename in uploaded_files:
            df = get_dataframe_from_file(filename)
            if df is not None:
                uploaded_files_data[filename] = df.head(100).to_html(classes='table table-striped table-hover', table_id=f'table-{filename}', escape=False)
        
        # Get available models
        models = []
        if OllamaService.is_server_running():
            models = OllamaService.get_available_models()
        
        return render_template('index.html', 
                             models=models,
                             uploaded_files_data=uploaded_files_data,
                             ai_response=response,
                             query_results_html=query_results_html)
    
    except Exception as e:
        flash(f'Error processing query: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
