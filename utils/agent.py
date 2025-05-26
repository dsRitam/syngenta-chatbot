import asyncio
import json
import os
import pymysql
import re
from typing import List, Dict, TypedDict
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
import requests
import nest_asyncio
from utils.auth import get_access_controls
from dotenv import load_dotenv
from datetime import date


load_dotenv()

# nest_asyncio for compatibility
nest_asyncio.apply()

# Global Configurations
BASE_URL = os.getenv("BEDROCK_BASE_URL")
API_KEY = os.getenv("BEDROCK_API_KEY")

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT")),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

chat_history = []
MAX_HISTORY = int(os.getenv("MAX_CHAT_HISTORY",10))
MAX_RETRIES = int(os.getenv("MAX_SQL_RETRIES",3))

# Initialize MySQL database connection
try:
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
    )
except Exception as e:
    print(f"[ERROR] Failed to connect to MySQL: {str(e)}")
    db = None

# Initialize embeddings
class BedrockEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text: str) -> List[float]:
        return get_bedrock_embedding(text)
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

embedding = BedrockEmbeddings()

# Load vectorstores
try:
    table_vectorstore = FAISS.load_local(
        "assets/table_vectorstore",
        embedding,
        allow_dangerous_deserialization=True
    )
    faiss_vectorstore = FAISS.load_local(
        "assets/faiss_vectorstore",
        embedding,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"[ERROR] Failed to load vectorstores: {str(e)}")
    table_vectorstore = None
    faiss_vectorstore = None

# <------------------------------------------------------BEDROCK APIs Function------------------------------------------------------------------->

# Claude 3.5 sonnet function
def call_bedrock_claude_3_5_sonnet(prompt: str, model_id: str = "claude-3.5-sonnet", max_tokens: int = 1500, temperature: float = 0.2) -> str:
    payload = {
        "api_key": API_KEY,
        "prompt": prompt,
        "model_id": model_id,
        "model_params": {
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if "response" in result and "content" in result["response"] and len(result["response"]["content"]) > 0:
            raw_response = result["response"]["content"][0]["text"]
            # print(f"[DEBUG] Bedrock Sonnet raw response: {raw_response[:500]}...")
            return raw_response
        else:
            raise ValueError("Unexpected response format from Bedrock API")
    except Exception as e:
        raise Exception(f"Error calling Bedrock API: {str(e)}")

# Claude 3 haiku function
def call_bedrock_claude_3_haiku(prompt: str, model_id: str = "claude-3-haiku", max_tokens: int = 500, temperature: float = 0.2) -> str:
    payload = {
        "api_key": API_KEY,
        "prompt": prompt,
        "model_id": model_id,
        "model_params": {
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if "response" in result and "content" in result["response"] and len(result["response"]["content"]) > 0:
            raw_response = result["response"]["content"][0]["text"]
            # print(f"[DEBUG] Bedrock Haiku raw response: {raw_response[:500]}...")
            return raw_response
        else:
            raise ValueError("Unexpected response format from Bedrock API")
    except Exception as e:
        raise Exception(f"Error calling Bedrock API: {str(e)}")

# embedding llm function
def get_bedrock_embedding(text: str) -> List[float]:
    payload = {
        "api_key": API_KEY,
        "prompt": text,
        "model_id": "amazon-embedding-v2"
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            return result["response"]["embedding"]
        else:
            raise Exception(f"Bedrock API error: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        raise Exception(f"Failed to get embedding: {str(e)}")



# <----------------------------------------------------------- Helper Function ------------------------------------------------------------------->

def format_chat_history(history: List[tuple]) -> str:
    if not history:
        return "No previous interactions."
    history_str = "Previous interactions:\n"
    for i, (q, ans) in enumerate(history, 1):
        history_str += f"{i}. Question: {q}\n  Answer: {ans}\n"
    return history_str

def is_safe_sql(query: str) -> bool:
    unsafe_patterns = [
        r"\bALTER\b", r"\bDROP\b", r"\bDELETE\b", r"\bTRUNCATE\b",
        r"\bUPDATE\b", r"\bINSERT\b"
    ]
    pattern = re.compile("|".join(unsafe_patterns), re.IGNORECASE)
    return not bool(pattern.search(query))

def check_access_restrictions(query: str, user_role: str = "user") -> str:
    db_restrictions, _ = get_access_controls(user_role)
    if not db_restrictions:
        return None
    for column in db_restrictions:
        if column.lower() in query.lower():
            return column
    return None

def clean_sql_query(sql_query: str) -> str:
    if not sql_query:
        return ""
    # remove code fences or sql keyword
    query = re.sub(r'```sql\s*|```', '', sql_query, flags=re.IGNORECASE).strip()
    # Add semicolon if missing
    if query and not query.endswith(';'):
        query += ';'
    # print(f"[DEBUG] Raw query: {sql_query}")
    # print(f"[DEBUG] Cleaned query: {query}")
    return query

def execute_sql_query(query: str):
    try:
        if not is_safe_sql(query):
            return "Blocked: Unsafe SQL command detected."
        conn = pymysql.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except pymysql.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Execution Error: {str(e)}"

def retrieve_table_extra_data(query: str) -> List[str]:
    if table_vectorstore is None:
        return []
    results = table_vectorstore.similarity_search(query, k=3)
    return [doc.page_content for doc in results]


# <--------------------------------------------------------------- RAG Agent -------------------------------------------------------------------->

class RAGAgentState(TypedDict):
    question: str
    user_role: str
    document_info: str
    response: str

async def retrieve_documents(state: RAGAgentState) -> RAGAgentState:
    try:
        if faiss_vectorstore is None:
            state["document_info"] = "Error: Rag vectorstore not initialized."
            return state
        retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
        def docs2str(docs, user_role):
            _, pdf_restrictions = get_access_controls(user_role)
            filtered_docs = []
            for doc in docs:
                source_file = doc.metadata.get("source_file", "")
                if source_file not in pdf_restrictions:
                    filtered_docs.append(f"Citation: {source_file}\n{doc.page_content}")
                else:
                    return f"{user_role} do not have access to {source_file}"
            return "\n\n".join(filtered_docs)

        docs = retriever.invoke(state["question"])
        state["document_info"] = docs2str(docs, state["user_role"])
        # print(f"[DEBUG] Document Info: {state['document_info'][:500]}...")
    except Exception as e:
        state["document_info"] = f"Error in document retrieval: {str(e)}"
    return state

async def generate_rag_response(state: RAGAgentState) -> RAGAgentState:
    try:
        template = '''
            Answer the question based only on the following context:
            {context}

            Chat History:
            {chat_history}

            Question: {question}

            Instructions:
            - Provide a concise answer (max 200 words) based on the context.
            - If context is insufficient, state so clearly.

            Answer:
        '''
        prompt = ChatPromptTemplate.from_template(template)
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: state["document_info"]),
                "chat_history": RunnableLambda(lambda x: format_chat_history(chat_history)),
                "question": RunnableLambda(lambda x: state["question"])
            }
            | prompt
            | RunnableLambda(lambda x: call_bedrock_claude_3_haiku(x.to_string()))
            | StrOutputParser()
        )
        response = await rag_chain.ainvoke(state["question"])
        state["response"] = response
        chat_history.append((state["question"], response))
        if len(chat_history) > MAX_HISTORY:
            chat_history.pop(0)
        print(f"[DEBUG] RAG Response: {response}")
    except Exception as e:
        state["response"] = f"Error generating response: {str(e)}"
    return state

def create_rag_agent_graph():
    workflow = StateGraph(RAGAgentState)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_rag_response", generate_rag_response)
    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_rag_response")
    workflow.add_edge("generate_rag_response", END)
    return workflow.compile()

async def run_rag_agent(question: str, user_role: str = "admin") -> str:
    agent_graph = create_rag_agent_graph()
    state = {
        "question": question,
        "user_role": user_role,
        "document_info": "",
        "response": ""
    }
    result = await agent_graph.ainvoke(state)
    return result["response"]


# <------------------------------------------------------------ SQL Ageent ----------------------------------------------------------------------->

class SQLAgentState(TypedDict):
    question: str
    user_role: str
    sql_query: str
    sql_result: str
    reasoning: str
    response: str
    retries: int
    exploratory_data: Dict[str, str]

async def generate_sql_query(state: SQLAgentState) -> SQLAgentState:
    try:
        table_info = db.get_table_info() if db else "Unknown schema; assume supply chain database with tables: supplychain (id, order_date, Sales, region, product, department_name, customer_state, profit_margin), customers (id, name, region)."
        print(f"[DEBUG] Table Info: {table_info}")
        table_extra_data = retrieve_table_extra_data(state["question"]) or []
        table_extra_data = "\n".join(table_extra_data)

        # Exploratory queries
        exploratory_data = {}
        
        # List tables
        table_query = "SHOW TABLES;"
        table_result = execute_sql_query(table_query)
        exploratory_data["tables"] = str([row[0] for row in table_result]) if not isinstance(table_result, str) else table_result
        print(f"[DEBUG] Exploratory Tables: {exploratory_data['tables']}")

        # Schema and data for relevant tables
        relevant_tables = ["supplychain", "customers"]
        for table in relevant_tables:
            if table.lower() in exploratory_data["tables"].lower():
                # Table schema
                describe_query = f"DESCRIBE {table};"
                describe_result = execute_sql_query(describe_query)
                exploratory_data[f"{table}_schema"] = str(describe_result) if not isinstance(describe_result, str) else describe_result

                # Sample data
                sample_query = f"SELECT * FROM {table} LIMIT 5;"
                sample_result = execute_sql_query(sample_query)
                exploratory_data[f"{table}_sample"] = str(sample_result) if not isinstance(sample_result, str) else sample_result

                # Distinct values
                key_columns = ["region", "product", "department_name"] if table == "supplychain" else ["region"]
                for col in key_columns:
                    distinct_query = f"SELECT DISTINCT {col} FROM {table} LIMIT 10;"
                    distinct_result = execute_sql_query(distinct_query)
                    exploratory_data[f"{table}_{col}_distinct"] = str(distinct_result) if not isinstance(distinct_result, str) else distinct_result

        state["exploratory_data"] = exploratory_data
        print(f"[DEBUG] Exploratory Data: {json.dumps(exploratory_data, indent=2)}")

        # Generate query
        prompt = '''
            You are a MySQL expert with autonomous reasoning capabilities. Generate a syntactically correct MySQL SELECT query to answer the question, using the schema, metadata, and exploratory data. Validate the query for correctness before returning.

            Schema:
            {table_info}

            Metadata:
            {extra_data}

            Exploratory Data:
            {exploratory_data}

            Chat History:
            {chat_history}

            Question: {question}

            Instructions:
            - Generate a query compatible with MySQL.
            - Map question terms to schema (e.g., "sales" to `Sales`, "orders" to `supplychain`).
            - Use exploratory data to validate tables/columns (e.g., check `supplychain_region_distinct` for regions).
            - Avoid `LIMIT` for total aggregations (e.g., "total sales").
            - Use table aliases (e.g., `s` for supplychain).
            - For date ranges, assume current date is {current_date} (e.g., last quarter = 3 months prior).
            - If unsure, use exploratory data to infer correct columns/tables.
            - Return only the query, no explanations.
            - Examples:
              Question: "Total sales amount for all orders"
              Query: SELECT SUM(s.Sales) AS total_sales FROM supplychain s;
              Question: "Products with highest profit margin"
              Query: SELECT s.product, s.profit_margin FROM supplychain s ORDER BY s.profit_margin DESC LIMIT 10;
              Question: "Sales in Southwest last quarter"
              Query: SELECT SUM(s.Sales) FROM supplychain s WHERE s.region = 'Southwest' AND s.order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH);

            Query:
        '''
        formatted_prompt = prompt.format(
            table_info=table_info,
            extra_data=table_extra_data,
            exploratory_data=json.dumps(exploratory_data),
            chat_history=format_chat_history(chat_history),
            question=state["question"],
            current_date=date.today().strftime('%Y-%m-%d')
        )
        sql_query = call_bedrock_claude_3_5_sonnet(formatted_prompt)
        cleaned_query = clean_sql_query(sql_query)
        if not cleaned_query:
            raise ValueError("Empty or invalid SQL query after cleaning")
        state["sql_query"] = cleaned_query
        state["reasoning"] = "Generated SQL query using Claude 3.5 Sonnet with exploratory data."
        print(f"[DEBUG] Generated SQL Query: {cleaned_query}")
    except Exception as e:
        state["sql_query"] = ""
        state["reasoning"] = f"Error generating SQL query: {str(e)}"
        state["response"] = f"Failed to generate query: {str(e)}. Please check schema or question clarity."
    return state

async def validate_and_execute_sql(state: SQLAgentState) -> SQLAgentState:
    if not state["sql_query"]:
        state["sql_result"] = "Error: No valid SQL query generated."
        state["reasoning"] = "No query to execute."
        return state

    restricted_column = check_access_restrictions(state["sql_query"], state["user_role"])
    if restricted_column:
        state["sql_result"] = f"Error: The column {restricted_column} is restricted for {state['user_role']}."
        state["reasoning"] = f"Query blocked due to restricted column: {restricted_column}"
        return state

    result = execute_sql_query(state["sql_query"])
    state["sql_result"] = result
    state["reasoning"] = "Executed SQL query successfully."
    print(f"[DEBUG] SQL Result: {result}")

    if isinstance(result, str) and result.startswith("SQL Error") and state["retries"] < MAX_RETRIES:
        state["retries"] += 1
        return await reason_and_correct_sql(state)
    return state

async def reason_and_correct_sql(state: SQLAgentState) -> SQLAgentState:
    try:
        table_info = db.get_table_info() if db else "Unknown schema; assume supply chain database with tables: supplychain (id, order_date, Sales, region, product, department_name, customer_state, profit_margin), customers (id, name, region)."
        prompt = """
            The SQL query failed with error: {sql_error}

            Failed Query: {sql_query}

            Question: {question}

            Schema: {table_info}

            Exploratory Data: {exploratory_data}

            Instructions:
            - Analyze the error and correct the MySQL query.
            - Ensure optimized, and MySQL-compatible.
            - Use exploratory data to validate columns/tables.
            - Return only the corrected query.

            Corrected Query:
        """
        formatted_prompt = prompt.format(
            sql_error=state["sql_result"],
            sql_query=state["sql_query"],
            question=state["question"],
            table_info=table_info,
            exploratory_data=json.dumps(state["exploratory_data"])
        )
        corrected_query = call_bedrock_claude_3_5_sonnet(formatted_prompt)
        cleaned_query = clean_sql_query(corrected_query)
        if not cleaned_query:
            raise ValueError("Empty or invalid corrected SQL query")
        state["sql_query"] = cleaned_query
        state["reasoning"] = f"Corrected SQL query after error (retry {state['retries']}): {state['sql_result']}"
        print(f"[DEBUG] Corrected SQL Query: {cleaned_query}")

        result = execute_sql_query(cleaned_query)
        state["sql_result"] = result
        print(f"[DEBUG] Corrected SQL Result: {result}")
    except Exception as e:
        state["reasoning"] = f"Error correcting SQL query: {str(e)}"
        state["response"] = f"Failed to correct query: {str(e)}. Please check schema or question clarity."
    return state

async def generate_sql_response(state: SQLAgentState) -> SQLAgentState:
    try:
        prompt = """
            Answer the question based on the SQL query and result (max 200 words).

            Question: {question}
            Query: {sql_query}
            Result: {sql_result}
            Reasoning: {reasoning}
            Chat History: {chat_history}

            Instructions:
            - Summarize results clearly.
            - If error, explain briefly and suggest next steps.
            - Use history context if relevant.

            Answer:
        """
        formatted_prompt = prompt.format(
            question=state["question"],
            sql_query=state["sql_query"],
            sql_result=str(state["sql_result"]),
            reasoning=state["reasoning"],
            chat_history=format_chat_history(chat_history)
        )
        response = call_bedrock_claude_3_haiku(formatted_prompt)
        state["response"] = response
        chat_history.append((state["question"], response))
        if len(chat_history) > MAX_HISTORY:
            chat_history.pop(0)
        print(f"[DEBUG] SQL Response: {response}")
    except Exception as e:
        state["response"] = f"Error generating response: {str(e)}"
    return state

def create_sql_agent_graph():
    workflow = StateGraph(SQLAgentState)
    workflow.add_node("generate_sql_query", generate_sql_query)
    workflow.add_node("validate_and_execute_sql", validate_and_execute_sql)
    workflow.add_node("generate_sql_response", generate_sql_response)
    workflow.set_entry_point("generate_sql_query")
    workflow.add_edge("generate_sql_query", "validate_and_execute_sql")
    workflow.add_edge("validate_and_execute_sql", "generate_sql_response")
    workflow.add_edge("generate_sql_response", END)
    return workflow.compile()

async def run_sql_agent(question: str, user_role: str = "admin") -> str:
    agent_graph = create_sql_agent_graph()
    state = {
        "question": question,
        "user_role": user_role,
        "sql_query": "",
        "sql_result": "",
        "reasoning": "",
        "response": "",
        "retries": 0,
        "exploratory_data": {}
    }
    result = await agent_graph.ainvoke(state)
    return result["response"]


# <--------------------------------------------------------------- Hybrid Agent ----------------------------------------------------------------->

class HybridAgentState(TypedDict):
    question: str
    user_role: str
    database_data: str
    document_info: str
    extracted_conditions: Dict
    sql_query: str
    sql_result: str
    policy_summary: str
    data_summary: str
    response: str
    agent_memory: List[Dict[str, str]]

async def parallel_retrieval(state: HybridAgentState) -> HybridAgentState:
    async def get_database_data():
        try:
            return await run_sql_agent(state["question"], state["user_role"])
        except Exception as e:
            return f"Error in database retrieval: {str(e)}"
    async def get_document_info():
        try:
            return await run_rag_agent(state["question"], state["user_role"])
        except Exception as e:
            return f"Error in document retrieval: {str(e)}"
    database_task, document_task = await asyncio.gather(
        get_database_data(),
        get_document_info(),
        return_exceptions=True
    )
    state["database_data"] = database_task if not isinstance(database_task, Exception) else f"Error: {str(database_task)}"
    state["document_info"] = document_task if not isinstance(document_task, Exception) else f"Error: {str(document_task)}"
    state["agent_memory"].append({
        "question": state["question"],
        "database_data": state["database_data"],
        "document_info": state["document_info"]
    })
    print(f"[DEBUG] Database Data: {state['database_data']}")
    print(f"[DEBUG] Document Info: {state['document_info'][:500]}...")
    return state

async def extract_conditions(state: HybridAgentState) -> HybridAgentState:
    try:
        prompt = """
            Extract relevant constraints from policy and database data in JSON format.

            Question: {question}
            Policy: {document_info}
            Database Data: {database_data}

            Instructions:
            - Return constraints as JSON (e.g., {{"condition": "region = 'Southwest'", "rule": "sales data"}}).
            - Return empty {} if no constraints.

            Constraints:
        """
        formatted_prompt = prompt.format(
            question=state["question"],
            document_info=state["document_info"],
            database_data=state["database_data"]
        )
        response = call_bedrock_claude_3_haiku(formatted_prompt)
        state["extracted_conditions"] = json.loads(response)
        print(f"[DEBUG] Extracted Conditions: {state['extracted_conditions']}")
    except Exception as e:
        state["extracted_conditions"] = {}
    return state

async def parallel_policy_sql(state: HybridAgentState) -> HybridAgentState:
    async def generate_policy_summary():
        try:
            prompt = """
                Summarize the policy relevant to the question (max 100 words).

                Question: {question}
                Policy: {document_info}
                Conditions: {conditions}

                Summary:
            """
            formatted_prompt = prompt.format(
                question=state["question"],
                document_info=state["document_info"],
                conditions=json.dumps(state["extracted_conditions"])
            )
            return call_bedrock_claude_3_haiku(formatted_prompt)
        except Exception as e:
            return f"Error in policy summary: {str(e)}"
    async def generate_sql():
        try:
            table_info = db.get_table_info() if db else "Unknown schema: supplychain (id, order_date, Sales, region, product, department_name, customer_state, profit_margin)."
            exploratory_data = {}
            table_query = "SHOW TABLES;"
            table_result = execute_sql_query(table_query)
            exploratory_data["tables"] = str([row[0] for row in table_result]) if not isinstance(table_result, str) else table_result

            if "supplychain" in exploratory_data["tables"].lower():
                describe_query = "DESCRIBE supplychain;"
                exploratory_data["supplychain_schema"] = str(execute_sql_query(describe_query))
                sample_query = "SELECT * FROM supplychain LIMIT 5;"
                exploratory_data["supplychain_sample"] = str(execute_sql_query(sample_query))
                for col in ["region", "product"]:
                    distinct_query = f"SELECT DISTINCT {col} FROM supplychain LIMIT 10;"
                    exploratory_data[f"supplychain_{col}_distinct"] = str(execute_sql_query(distinct_query))

            prompt = """
                Generate a MySQL SELECT query based on the question, conditions, and exploratory data.

                Question: {question}
                Conditions: {conditions}
                Schema: {table_info}
                Exploratory Data: {exploratory_data}

                Instructions:
                - Generate only a SELECT query.
                - Use conditions and exploratory data to refine the query.
                - Avoid `LIMIT` for total aggregations, trend analysis.
                - Example: SELECT SUM(s.Sales) FROM supplychain s WHERE s.region = 'Southwest';

                Query:
            """
            formatted_prompt = prompt.format(
                question=state["question"],
                conditions=json.dumps(state["extracted_conditions"]),
                table_info=table_info,
                exploratory_data=json.dumps(exploratory_data)
            )
            sql_query = call_bedrock_claude_3_5_sonnet(formatted_prompt)
            cleaned_query = clean_sql_query(sql_query)
            if not cleaned_query:
                raise ValueError("Empty or invalid SQL query")
            return cleaned_query
        except Exception as e:
            return f"Error in SQL generation: {str(e)}"
    policy_task, sql_task = await asyncio.gather(
        generate_policy_summary(),
        generate_sql(),
        return_exceptions=True
    )
    state["policy_summary"] = policy_task if not isinstance(policy_task, Exception) else f"Error: {str(policy_task)}"
    state["sql_query"] = sql_task if not isinstance(sql_task, str) or not sql_task.startswith("Error") else ""
    print(f"[DEBUG] Policy Summary: {state['policy_summary']}")
    print(f"[DEBUG] Hybrid SQL Query: {state['sql_query']}")
    return state

async def execute_hybrid_sql(state: HybridAgentState) -> HybridAgentState:
    if not state["sql_query"]:
        state["sql_result"] = "Error: No valid SQL query generated."
        return state
    restricted_column = check_access_restrictions(state["sql_query"], state["user_role"])
    if restricted_column:
        state["sql_result"] = f"Error: The column {restricted_column} is restricted for {state['user_role']}."
        return state
    try:
        sql_result = execute_sql_query(state["sql_query"])
        state["sql_result"] = sql_result
        print(f"[DEBUG] Hybrid SQL Result: {sql_result}")
    except Exception as e:
        state["sql_result"] = f"Error in SQL execution: {str(e)}"
    return state

async def analyze_results(state: HybridAgentState) -> HybridAgentState:
    try:
        prompt = """
            Analyze SQL results against policy conditions.

            Question: {question}
            Conditions: {conditions}
            SQL Result: {sql_result}

            Instructions:
            - Provide concise analysis (max 100 words) of data-policy alignment.

            Analysis:
        """
        formatted_prompt = prompt.format(
            question=state["question"],
            conditions=json.dumps(state["extracted_conditions"]),
            sql_result=str(state["sql_result"])
        )
        state["data_summary"] = call_bedrock_claude_3_haiku(formatted_prompt)
        print(f"[DEBUG] Data Summary: {state['data_summary']}")
    except Exception as e:
        state["data_summary"] = f"Error in result analysis: {str(e)}"
    return state

async def combine_hybrid_results(state: HybridAgentState) -> HybridAgentState:
    try:
        prompt = """
            Combine policy and database insights to answer the question.

            Question: {question}
            Policy Summary: {policy_summary}
            Data Summary: {data_summary}
            Chat History: {chat_history}

            Instructions:
            - Provide a clear, concise answer (max 200 words).
            - Address missing data or policy issues.

            Answer:
        """
        formatted_prompt = prompt.format(
            question=state["question"],
            policy_summary=state["policy_summary"],
            data_summary=state["data_summary"],
            chat_history=format_chat_history(chat_history)
        )
        response = call_bedrock_claude_3_haiku(formatted_prompt)
        state["response"] = response
        state["agent_memory"].append({"question": state["question"], "response": response})
        chat_history.append((state["question"], response))
        if len(chat_history) > MAX_HISTORY:
            chat_history.pop(0)
        print(f"[DEBUG] Hybrid Response: {response}")
    except Exception as e:
        state["response"] = f"Error in combining results: {str(e)}"
    return state

def create_hybrid_agent_graph():
    workflow = StateGraph(HybridAgentState)
    workflow.add_node("parallel_retrieval", parallel_retrieval)
    workflow.add_node("extract_conditions", extract_conditions)
    workflow.add_node("parallel_policy_sql", parallel_policy_sql)
    workflow.add_node("execute_hybrid_sql", execute_hybrid_sql)
    workflow.add_node("analyze_results", analyze_results)
    workflow.add_node("combine_hybrid_results", combine_hybrid_results)
    workflow.set_entry_point("parallel_retrieval")
    workflow.add_edge("parallel_retrieval", "extract_conditions")
    workflow.add_edge("extract_conditions", "parallel_policy_sql")
    workflow.add_edge("parallel_policy_sql", "execute_hybrid_sql")
    workflow.add_edge("execute_hybrid_sql", "analyze_results")
    workflow.add_edge("analyze_results", "combine_hybrid_results")
    workflow.add_edge("combine_hybrid_results", END)
    return workflow.compile()

async def run_hybrid_agent(question: str, user_role: str = "admin") -> str:
    agent_graph = create_hybrid_agent_graph()
    state = {
        "question": question,
        "user_role": user_role,
        "database_data": "",
        "document_info": "",
        "extracted_conditions": {},
        "sql_query": "",
        "sql_result": "",
        "policy_summary": "",
        "data_summary": "",
        "response": "",
        "agent_memory": []
    }
    result = await agent_graph.ainvoke(state)
    return result["response"]


# <------------------------------------------------ First Point of Contact Decision Tool ----------------------------------------------------->
async def decide_tool(question: str, user_role: str = "admin") -> str:
    from utils.metrics import log_metrics
    import time

    start_time = time.time()
    prompt = """
        Select the tool for the question:
        - RAG (for only policies, guidelines)
        - SQL (for only supply chain data like sales, inventory)
        - HYBRID (for both policy/guidelines/framework and data)
        - NONE (if neither applies)

        Question: {question}

        Instructions:
        - Return only the tool name (RAG, SQL, HYBRID, NONE).

        Tool:
    """
    formatted_prompt = prompt.format(question=question)
    try:
        selected_tool = call_bedrock_claude_3_haiku(formatted_prompt).strip().upper()
    except Exception as e:
        selected_tool = "NONE"
        print(f"[DEBUG] Error in tool selection: {str(e)}")
    print(f"[DEBUG] Selected Tool: {selected_tool}")

    
    query_type = selected_tool

    if selected_tool == "SQL":
        response = await run_sql_agent(question, user_role)
    elif selected_tool == "RAG":
        response = await run_rag_agent(question, user_role)
    elif selected_tool == "HYBRID":
        response = await run_hybrid_agent(question, user_role)
    else:
        query_type = "NONE"
        prompt = """
            Answer based on general knowledge and chat history.

            Question: {question}
            Chat History: {history}

            Answer:
        """
        formatted_prompt = prompt.format(
            question=question,
            history=format_chat_history(chat_history)
        )
        response = call_bedrock_claude_3_5_sonnet(formatted_prompt).strip()
        chat_history.append((question, response))
        if len(chat_history) > MAX_HISTORY:
            chat_history.pop(0)


    response_time = time.time() - start_time
    log_metrics(query_type, question, response_time, token_usage=None, accuracy=None)
    return response