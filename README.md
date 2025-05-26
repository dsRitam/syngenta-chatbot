# Supply Chain AI Agent

Syngenta Chatbot is a Python-based Agentic AI web application designed to assist users in querying supply chain data and policies. Built using Streamlit, it integrates with MySQL and SQLite databases, leverages FAISS vector stores for document retrieval, and utilizes AWS Bedrock APIs for natural language processing. The application supports role-based access control, web search capabilities, and performance monitoring for administrators.

## Features

- **Role-Based Authentication**: Sign up and sign in with roles (`admin`, `manager`, `user`) to access features based on permissions.
- **Supply Chain Querying**: Ask questions about supply chain data (e.g., sales, inventory) and policies using three agents:
  - **RAG Agent**: Retrieves and summarizes policy documents.
  - **SQL Agent**: Executes safe SQL queries on a MySQL database for data insights.
  - **Hybrid Agent**: Combines policy and data insights for comprehensive answers.
- **Web Search**: Perform web searches via Google Serper API, with results summarized by AWS Bedrock's Claude 3 Haiku model and Claude 3.5 sonnet.
- **Embeddings**: Uses the amazon-embedding-v2 model via AWS Bedrock to generate embeddings for FAISS vector stores, enabling semantic search for documents and table metadata
- **Access Control Management**: Admins can manage restrictions on database columns and PDFs for different roles.
- **Performance Metrics**: Admins can view query statistics, including total queries, average response times, and query logs, with visualizations using Plotly.
- **Chat History**: Persistent chat history for reviewing past interactions.
- **Secure and Scalable**: Uses environment variables for sensitive configurations and supports MySQL for scalable data storage.

## Project Structure

```
SYNGENTA_CHATBOT/
├── assets/
│   ├── faiss_vectorstore/        # FAISS vector store for document retrieval
│   └── table_vectorstore/        # FAISS vector store for table metadata
├── database/
│   ├── init_db.py                # Script to initialize SQLite database
│   └── local_db.sqlite           # SQLite database for user data and metrics
├── utils/
|   ├── agent.py                  # Core logic for RAG, SQL, and Hybrid agents
│   ├── auth.py                   # Authentication and access control logic
│   ├── metrics.py                # Metrics logging and analysis
│   └── web_search.py             # Web search functionality
├── .env                          # Environment variables (not tracked in git)
├── app.py                        # Main Streamlit application
├── README.md                     # Readme file 
└── requirements.txt              # Python dependencies

```

## Prerequisites

- Python 3.8 or higher
- MySQL server (for supply chain data)
- Google Serper API key (for web search)
- AWS Bedrock API key (for NLP and embeddings)
- SQLite (for local database)

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/dsRitam/syngenta-chatbot.git
   cd syngenta-chatbot
   ```

2. **Install Dependencies**: Create a virtual environment and install the required packages:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**: Create a `.env` file in the root directory and add the following variables:

   ```plaintext
   GOOGLE_SERPER_API_KEY=<your-serper-api-key>
   BEDROCK_BASE_URL=<aws-bedrock-base-url>
   BEDROCK_API_KEY=<aws-bedrock-api-key>
   MYSQL_HOST=<mysql-host>
   MYSQL_PORT=<mysql-port>
   MYSQL_USER=<mysql-user>
   MYSQL_PASSWORD=<mysql-password>
   MYSQL_DATABASE=<mysql-database>
   MAX_CHAT_HISTORY=10
   MAX_SQL_RETRIES=3
   ```

   Replace the placeholders with your actual values.

4. **Initialize the SQLite Database**: Run the database initialization script to create the SQLite database and tables:

   ```bash
   python database/init_db.py
   ```

   This creates `local_db.sqlite` with pre-seeded admin credentials (`email: admin@email.com`, `password: admin@123`).

5. **Set Up MySQL Database**: Ensure your MySQL server is running and contains the necessary supply chain data. Update the schema as needed based on your data requirements.

6. Store FAISS vector store files in the `assets/` directory (`faiss_vectorstore` for document retrieval, `table_vectorstore` for table metadata)—these are not covered in this project file.

7. **Run the Application**: Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   Access the app at `http://localhost:8501` in your browser.

## Usage

1. **Sign In or Sign Up**:

   - Use the pre-seeded admin account (`admin@email.com`, `admin@123`) or sign up as a new user with a role (`manager` or `user`).
   - Roles determine access levels (e.g., admins can manage access controls).

2. **Ask Questions**:

   - Use the chatbot interface to ask questions about supply chain data or policies (e.g., "Total sales of the company").
   - The app automatically routes queries to the appropriate agent (RAG, SQL, or Hybrid).

3. **Web Search**:

   - Use the sidebar to perform web searches for external information, with a 5-second cooldown between searches.

4. **Admin Dashboard** (Admin Only):

   - **Chatbot Tab**: Same as the user chatbot interface.
   - **Access Control Tab**: Manage restrictions on database columns and PDFs for `manager` and `user` roles.
   - **Performance Metrics Tab**: View query statistics, visualizations (bar and line charts), and download query logs as CSV or Markdown.

5. **Logout**:

   - Use the logout button to end your session and clear chat history.

## Role-Based Access

- **Admin**:
  - Full access to all features, including the admin dashboard.
  - Can manage access controls and view performance metrics.
- **Manager**:
  - Access to the chatbot and web search.
  - Restricted columns and PDFs apply based on admin settings.
- **User**:
  - Similar to `manager` but with potentially stricter restrictions.

## Troubleshooting

- **Database Errors**:
  - Ensure `local_db.sqlite` exists in the `database/` directory. Run `python database/init_db.py` if missing.
  - Verify MySQL connection details in `.env`.
- **API Issues**:
  - Check your Google Serper and AWS Bedrock API keys in `.env`.
  - Ensure your AWS Bedrock region and credentials are correctly configured.
- **Performance Metrics**:
  - Metrics require queries to be logged. Start using the chatbot to populate the metrics table.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or support, please contact : ds.ritam25@gmail.com.

---