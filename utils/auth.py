import sqlite3
from typing import List, Tuple, Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
import os
import requests
import json
import pymysql
from dotenv import load_dotenv



load_dotenv()

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

# initialize mysql connection
try:
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
    )
except Exception as e:
    print(f"[ERROR] Failed to connect to MySQL: {str(e)}")
    db = None


def execute_sql_query_auth(query: str):
    try:
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

# initialize embeddings 
BASE_URL = os.getenv("BEDROCK_BASE_URL")
API_KEY = os.getenv("BEDROCK_API_KEY")

def get_bedrock_embedding(text: str):
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

class BedrockEmbeddings:
    def embed_documents(self, texts: List[str]):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text: str):
        return get_bedrock_embedding(text)
    def __call__(self, text: str):
        return self.embed_query(text)


embedding = BedrockEmbeddings()

# load faiss_vectorstore
try:
    faiss_vectorstore = FAISS.load_local(
        "assets/faiss_vectorstore",
        embedding,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"[ERROR] Failed to load faiss_vectorstore: {str(e)}")
    faiss_vectorstore = None

def authenticate_user(email: str, password: str) -> Optional[str]:
    try:
        conn = sqlite3.connect("database/local_db.sqlite")
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE email = ? AND password = ?", (email, password))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        print(f"[ERROR] Authentication failed: {str(e)}")
        return None

def signup_user(username: str, email: str, password: str, role: str, country: str) -> bool:
    try:
        conn = sqlite3.connect("database/local_db.sqlite")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE email = ?", (email,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            return False
        cursor.execute(
            "INSERT INTO users (username, email, password, role, country) VALUES (?, ?, ?, ?, ?)",
            (username, email, password, role, country)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Signup failed: {str(e)}")
        return False

def get_access_controls(role: str) -> Tuple[List[str], List[str]]:
    """Get restricted columns and PDFs for a role."""
    try:
        conn = sqlite3.connect("database/local_db.sqlite")
        cursor = conn.cursor()
        # Database restrictions
        cursor.execute("SELECT restricted_column FROM db_access_control WHERE role = ?", (role,))
        db_restrictions = sorted(set(row[0] for row in cursor.fetchall()))
        # PDF restrictions
        cursor.execute("SELECT restricted_pdf FROM pdf_access_control WHERE role = ?", (role,))
        pdf_restrictions = sorted(set(row[0] for row in cursor.fetchall()))
        conn.close()
        # print(f"[DEBUG] Access controls for {role}: DB={db_restrictions}, PDFs={pdf_restrictions}")
        return db_restrictions, pdf_restrictions
    except Exception as e:
        print(f"[ERROR] Failed to get access controls: {str(e)}")
        return [], []

def get_all_columns_and_pdfs() -> Tuple[List[str], List[str]]:
    """Fetch all columna from mysql db and all pdf from vectorstore."""
    columns = []
    pdfs = []
    
    # Fetch columns from MySQL database
    try:
        if db:
            # Step 1: Fetch all table names
            query="SHOW TABLES;"
            tables_result = execute_sql_query_auth(query)
            
            # Extract table names correctly
            table_names = [row[0] for row in tables_result if isinstance(row, tuple) and row]
            print(f"[DEBUG] Tables found: {table_names}")

            # Step 2: Run DESCRIBE on each table to collect columns
            columns = []

            for table in table_names:
                try:
                    query = f"DESCRIBE {table};"
                    result = execute_sql_query_auth(query)

                    # Extract column names and append to list
                    for row in result:
                        if isinstance(row, tuple) and row:
                            columns.append(row[0])

                except pymysql.err.ProgrammingError as e:
                    print(f"[ERROR] Failed to DESCRIBE {table}: {str(e)}")

            columns =  sorted(columns) 
            print(f'DEBUG: {columns}')
        else:
            columns = ['Type']
        # print(f"[DEBUG] Available columns: {columns}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch columns from MySQL: {str(e)}")
        columns = ['Type']
        print(f"[DEBUG] Using fallback columns:")
    
    # Fetch PDFs from faiss_vectorstore 
    try:
        if faiss_vectorstore:
            docs = faiss_vectorstore.docstore._dict 
            pdf_set = set()
            for doc_id, doc in docs.items():
                source_file = doc.metadata.get("source_file", "")
                if source_file:
                    pdf_set.add(source_file)
            pdfs = sorted(list(pdf_set))
        else:
            pdfs = ["Anti-Counterfeit and Product Authenticity Policy.pdf", "Labor Standards.pdf", "IOT.pdf"]
        print(f"[DEBUG] Available PDFs: {pdfs}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch PDFs from faiss_vectorstore: {str(e)}")
        pdfs = ["Anti-Counterfeit and Product Authenticity Policy.pdf", "Labor Standards.pdf", "IOT.pdf"]
        print(f"[DEBUG] Using fallback PDFs: {pdfs}")
    
    return columns, pdfs