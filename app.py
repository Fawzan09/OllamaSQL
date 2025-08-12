from flask import Flask, render_template, request, redirect, url_for, session, flash
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
import uuid
from io import StringIO

# Constants
OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 10

# Flask app setup
app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-secret"

# In-memory server-side state store keyed by a per-session sid
_STATE_STORE: Dict[str, Dict[str, Any]] = {}


def get_sid() -> str:
    """Ensure a unique session id exists and return it."""
    if 'sid' not in session:
        session['sid'] = uuid.uuid4().hex
    return session['sid']


def get_state() -> Dict[str, Any]:
    """Return mutable per-session state dict stored server-side."""
    sid = get_sid()
    if sid not in _STATE_STORE:
        _STATE_STORE[sid] = {}
    return _STATE_STORE[sid]

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
        state = get_state()
        state.setdefault('uploaded_files', {})  # filename -> DataFrame
        state.setdefault('selected_df', None)
        state.setdefault('selected_df_file_name', None)
        state.setdefault('vector_indices_created', False)
        state.setdefault('embedding_model_name', None)
        state.setdefault('similarity_metric', 'cosine')
        state.setdefault('selected_columns', [])
        state.setdefault('model_name', None)
        state.setdefault('temperature', 0.7)

class SQLExecutor:
    """Handles SQL query execution"""
    @staticmethod
    def execute_query(query: str) -> str:
        try:
            state = get_state()
            if state.get('selected_df') is None:
                return json.dumps({"error": "No dataframe selected"})
            
            with duckdb.connect() as conn:
                # Register the in-memory DataFrame as 'selected_df'
                conn.register('selected_df', state['selected_df'])
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
        similarity_metric: str
    ) -> AIResponse:
        client = ollama.AsyncClient()
        state = get_state()
        df = state['selected_df']
        num_rows = df.shape[0]

        # Build detailed columns info
        columns_info_lines: List[str] = []
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

        # Send the prompt to the model and stream the response (aggregate to string)
        assistant_response_text = ""
        try:
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
            return AIResponse(sql_query=None, analysis=f"Error during assistant generation: {e}", sql_results=None)

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
                return AIResponse(sql_query=sql_query, analysis=f"Error during query embedding: {e}", sql_results=None)

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
        # Final analysis (aggregate)
        assistant_final_response = ""
        try:
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
            assistant_final_response = f"Error during final response generation: {e}"

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
class StreamlitUI:
    """Deprecated: Streamlit UI removed. Flask routes now provide the interface."""
    pass


class FlaskUI:
    """Flask route handlers are defined below."""
    pass


def create_vector_indices() -> Tuple[bool, str]:
    """Create embeddings and vector index in DuckDB. Returns (success, message)."""
    state = get_state()
    df = state.get('selected_df')
    selected_columns = state.get('selected_columns', [])
    embedding_model_name = state.get('embedding_model_name')
    if df is None:
        return False, "No dataframe selected."
    if not selected_columns:
        return False, "No columns selected for embedding."
    if not embedding_model_name:
        return False, "No embedding model selected."

    embeddings: List[List[float]] = []
    total = len(df)
    try:
        for idx, row in df.iterrows():
            row_texts = [str(row[col]) for col in selected_columns if pd.notnull(row[col])]
            combined_text = ' '.join(row_texts).strip() or " "
            embedding_response = ollama.embed(model=embedding_model_name, input=[combined_text])
            query_embedding = embedding_response['embedding'][0] if 'embedding' in embedding_response else embedding_response['embeddings'][0]
            embeddings.append([float(v) for v in query_embedding])
            # tiny sleep to keep server responsive
            if idx % 50 == 0:
                sleep(0.001)
    except Exception as e:
        return False, f"Error during embedding at index {idx}: {e}"

    if len(embeddings) != len(df):
        return False, f"Embeddings count ({len(embeddings)}) does not match rows ({len(df)})."

    df = df.copy()
    df['embedding'] = embeddings

    # Determine the embedding dimension
    embedding_length = len(embeddings[0]) if embeddings else 0

    try:
        conn = duckdb.connect()
        conn.execute("INSTALL 'vss';")
        conn.execute("LOAD 'vss';")

        dtype_mapping = {
            'object': 'VARCHAR',
            'int64': 'BIGINT',
            'float64': 'DOUBLE',
            'bool': 'BOOLEAN'
        }

        column_defs = []
        for col in df.columns:
            if col == 'embedding':
                column_defs.append(f"{col} FLOAT[{embedding_length}]")
            else:
                pandas_dtype = df[col].dtype.name
                duckdb_type = dtype_mapping.get(pandas_dtype, 'VARCHAR')
                column_defs.append(f"{col} {duckdb_type}")

        conn.execute("DROP TABLE IF EXISTS selected_df")
        conn.execute(f"""
            CREATE TABLE selected_df (
                {', '.join(column_defs)}
            )
        """)

        data = df.to_records(index=False).tolist()
        placeholders = ', '.join(['?' for _ in df.columns])
        conn.executemany(f"INSERT INTO selected_df VALUES ({placeholders})", data)
        similarity_metric = get_state().get('similarity_metric', 'cosine')
        conn.execute(f"""
            CREATE INDEX hnsw_idx ON selected_df USING HNSW (embedding) WITH (metric = '{similarity_metric}')
        """)
        conn.close()
    except Exception as e:
        return False, f"DuckDB error during index creation: {e}"

    state['selected_df'] = df
    state['vector_indices_created'] = True
    return True, "Vector indices created successfully."


# ---------------------- Flask Routes ----------------------

@app.before_request
def ensure_state():
    DataFrameManager.initialize_session_state()


@app.route('/', methods=['GET'])
def index():
    state = get_state()
    server_ok = OllamaService.is_server_running()
    available_models = OllamaService.get_available_models() if server_ok else []
    uploaded_files = list(state['uploaded_files'].keys())
    selected_df = state['selected_df']
    data_tabs = [(name, state['uploaded_files'][name].head(100)) for name in uploaded_files]
    ai_response: Optional[AIResponse] = state.get('last_ai_response')
    return render_template(
        'index.html',
        server_ok=server_ok,
        available_models=available_models,
        model_name=state.get('model_name'),
        embedding_model_name=state.get('embedding_model_name'),
        temperature=state.get('temperature', 0.7),
        similarity_metric=state.get('similarity_metric', 'cosine'),
        uploaded_files=uploaded_files,
        selected_file=state.get('selected_df_file_name'),
        selected_columns=state.get('selected_columns', []),
        data_tabs=data_tabs,
        has_index=state.get('vector_indices_created', False),
        ai_response=ai_response,
        query_text=state.get('last_query', '')
    )


@app.route('/upload', methods=['POST'])
def upload():
    state = get_state()
    files = request.files.getlist('csv_files')
    count_added = 0
    for f in files:
        if f and f.filename:
            try:
                content = f.read().decode('utf-8', errors='ignore')
                df = pd.read_csv(StringIO(content))
                state['uploaded_files'][f.filename] = df
                count_added += 1
            except Exception as e:
                flash(f"Failed to read {f.filename}: {e}", 'danger')
    if count_added:
        flash(f"Uploaded {count_added} file(s).", 'success')
    return redirect(url_for('index'))


@app.route('/select_file', methods=['POST'])
def select_file():
    state = get_state()
    fname = request.form.get('selected_file')
    if fname and fname in state['uploaded_files']:
        state['selected_df'] = state['uploaded_files'][fname]
        state['selected_df_file_name'] = fname
        state['vector_indices_created'] = False
        state['selected_columns'] = []
    return redirect(url_for('index'))


@app.route('/set_models', methods=['POST'])
def set_models():
    state = get_state()
    state['model_name'] = request.form.get('model_name') or None
    state['embedding_model_name'] = request.form.get('embedding_model_name') or None
    state['similarity_metric'] = request.form.get('similarity_metric') or 'cosine'
    try:
        state['temperature'] = float(request.form.get('temperature', '0.7'))
    except ValueError:
        state['temperature'] = 0.7
    flash('Model settings updated.', 'info')
    return redirect(url_for('index'))


@app.route('/set_columns', methods=['POST'])
def set_columns():
    state = get_state()
    cols = request.form.getlist('selected_columns')
    state['selected_columns'] = cols
    state['vector_indices_created'] = False
    flash('Selected columns updated.', 'info')
    return redirect(url_for('index'))


@app.route('/create_index', methods=['POST'])
def route_create_index():
    ok, msg = create_vector_indices()
    flash(msg, 'success' if ok else 'danger')
    return redirect(url_for('index'))


@app.route('/query', methods=['POST'])
def query_route():
    state = get_state()
    user_query = request.form.get('user_query', '').strip()
    state['last_query'] = user_query
    if not user_query:
        flash('Please enter a query.', 'warning')
        return redirect(url_for('index'))
    if not OllamaService.is_server_running():
        flash('Ollama server is not running.', 'danger')
        return redirect(url_for('index'))
    if state.get('selected_df') is None:
        flash('Please select a dataset first.', 'warning')
        return redirect(url_for('index'))
    if not state.get('vector_indices_created'):
        flash('Please create vector indices before querying.', 'warning')
        return redirect(url_for('index'))

    try:
        response = asyncio.run(
            OllamaService.get_ai_response(
                user_query,
                state.get('model_name') or '',
                state.get('embedding_model_name') or '',
                float(state.get('temperature', 0.7)),
                state.get('similarity_metric', 'cosine')
            )
        )
        state['last_ai_response'] = response
    except Exception as e:
        flash(f"Error during analysis: {e}", 'danger')
    return redirect(url_for('index'))


def run():
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == "__main__":
    run()
