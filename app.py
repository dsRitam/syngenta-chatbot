import streamlit as st
import asyncio
import time
from utils.auth import authenticate_user, signup_user, get_access_controls, get_all_columns_and_pdfs
from utils.metrics import get_metrics, calculate_stats
from utils.agent import decide_tool
from utils.web_search import web_search_and_process
from database.init_db import init_db
import sqlite3
import pandas as pd
import plotly.express as px
import datetime

# initialize SQLite database
init_db()

# streamlit app
st.set_page_config(page_title="Syngenta Chatbot", layout="wide")

# Session state for authentication
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "email" not in st.session_state:
    st.session_state.email = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Session state for editing_role in Access Control tab
if "editing_role" not in st.session_state:
    st.session_state.editing_role = None
# Session state to prevent rapid save clicks
if "last_save_time" not in st.session_state:
    st.session_state.last_save_time = 0
# Session state for web search
if "web_search_query" not in st.session_state:
    st.session_state.web_search_query = ""
if "web_search_result" not in st.session_state:
    st.session_state.web_search_result = ""
if "last_web_search_time" not in st.session_state:
    st.session_state.last_web_search_time = 0

def main():
    if not st.session_state.user_role:
        show_auth_page()
    else:
        # Sidebar: Web search feature
        with st.sidebar:
            st.subheader("🔍 Web Search")
            with st.expander("Perform a Web Search", expanded=False):
                query = st.text_input("Ask your question:", key="web_search_input")
                if st.button("Search", key="web_search_button"):
                    if query:
                        current_time = time.time()
                        if current_time - st.session_state.last_web_search_time < 5:  # 5-second cooldown
                            st.warning("Please wait a moment before searching again.")
                        else:
                            st.session_state.last_web_search_time = current_time
                            with st.spinner("Searching the web..."):
                                try:
                                    result = web_search_and_process(query)
                                    st.session_state.web_search_query = query
                                    st.session_state.web_search_result = result
                                except Exception as e:
                                    st.error(f"Web search failed: {str(e)}")
                if st.session_state.web_search_result:
                    st.write("**Search Result:**")
                    st.write(st.session_state.web_search_result)
                    if st.button("Clear", key="clear_web_search"):
                        st.session_state.web_search_query = ""
                        st.session_state.web_search_result = ""
                        st.rerun()

        # Sidebar: Chat history
        with st.sidebar:
            st.subheader("Chat History")
            if not st.session_state.chat_history:
                st.write("No chat history yet.")
            else:
                for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                    truncate_length = min(len(q), 50)
                    with st.expander(f"Q{i}: {q[:truncate_length]}..."):
                        st.write(f"**Question**: {q}")
                        st.write(f"**Answer**: {a}")
        
        if st.session_state.user_role == "admin":
            show_admin_dashboard()
        else:
            show_chatbot()

def show_auth_page():
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <img src="https://2.bp.blogspot.com/-NjnwHOaA_RM/XEvDcmc6KqI/AAAAAAAARyw/7MpTMpRadmwfqY9Ef8Ffqv_qIOCXH3a7QCLcBGAs/w1200-h630-p-k-no-nu/Sygenta.png"
             width="180" height="100" style="margin-right: 15px; margin-bottom: 20px;">
        <h1 style="margin: 0;">Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        # st.header("Sign In")
        email = st.text_input("Email", key="signin_email")
        password = st.text_input("Password", type="password", key="signin_password")
        if st.button("Sign In"):
            role = authenticate_user(email, password)
            if role:
                st.session_state.user_role = role
                st.session_state.email = email
                st.success(f"Success: Logged in as {role}")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        # st.header("Sign Up")
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        role = st.selectbox("Role", ["manager", "user"], key="signup_role")
        country = st.text_input("Country", key="signup_country")
        if st.button("Sign Up"):
            if signup_user(username, email, password, role, country):
                st.success("Success: User created successfully! Please sign in.")
            else:
                st.error("Email already exists or invalid data")

def show_chatbot():
    st.title("Supply Chain Chatbot")
    st.write(f"Logged in as: {st.session_state.user_role} ({st.session_state.email})")
    
    # Logout button
    if st.button("Logout"):
        st.session_state.user_role = None
        st.session_state.email = None
        st.session_state.chat_history = []
        st.session_state.web_search_query = ""  
        st.session_state.web_search_result = ""  
        st.session_state.last_web_search_time = 0  
        st.rerun()
    
    # Chat interface
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if question:
            with st.spinner("Processing..."):
                result = asyncio.run(decide_tool(question, st.session_state.user_role))
                st.session_state.chat_history.append((question, result))
            st.write("**Answer**:")
            st.write(result)

def show_admin_dashboard():
    st.title("Admin Dashboard")
    st.write(f"Logged in as: {st.session_state.user_role} ({st.session_state.email})")
    
    # Logout button
    if st.button("Logout"):
        st.session_state.user_role = None
        st.session_state.email = None
        st.session_state.chat_history = []
        st.session_state.editing_role = None
        st.session_state.last_save_time = 0
        st.session_state.web_search_query = ""  
        st.session_state.web_search_result = ""  
        st.session_state.last_web_search_time = 0  
        st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["Chatbot", "Access Control", "Performance Metrics"])
    
    with tab1:
        st.subheader("Chatbot")
        question = st.text_input("Ask a question:", key="admin_chat")
        if st.button("Submit", key="admin_submit"):
            if question:
                with st.spinner("Processing..."):
                    result = asyncio.run(decide_tool(question, st.session_state.user_role))
                    st.session_state.chat_history.append((question, result))
                st.write("**Answer**:")
                st.write(result)
    
    with tab2:
        st.subheader("Manage Access Controls")
        
        # Fetching available columns and PDFs
        with st.spinner("Fetching columns and PDFs..."):
            columns, pdfs = get_all_columns_and_pdfs()
        
        if not columns:
            st.error("No columns available to restrict. Check MySQL database connection.")
        if not pdfs:
            st.error("No PDFs available to restrict. Check faiss_vectorstore.")
        
        # Fetching current restrictions for all roles
        roles = ["manager", "user"]
        restrictions_data = []
        for role in roles:
            db_restrictions, pdf_restrictions = get_access_controls(role)
            restrictions_data.append({
                "Role": role,
                "Restricted Columns": ", ".join(db_restrictions) if db_restrictions else "None",
                "Restricted PDFs": ", ".join(pdf_restrictions) if pdf_restrictions else "None",
            })
        
        # Display current restrictions in a table
        st.write("Current Access Controls:")
        df = pd.DataFrame(restrictions_data)
        st.table(df)
        
        # Role selection for editing
        st.write("Select a role to edit access controls:")
        selected_role = st.selectbox("Role", roles, key="edit_role_select")
        
        if st.button("Edit Restrictions"):
            st.session_state.editing_role = selected_role
        
        # Show editing interface if a role is selected
        if st.session_state.editing_role:
            st.write(f"Editing restrictions for role: **{st.session_state.editing_role}**")
            
            # Fetch current restrictions for the selected role
            current_db_restrictions, current_pdf_restrictions = get_access_controls(st.session_state.editing_role)
            
            # Multi-select for columns
            new_db_restrictions = st.multiselect(
                "Select restricted columns",
                options=columns,
                default=current_db_restrictions,
                key=f"db_restrictions_{st.session_state.editing_role}"
            )
            
            # Multi-select for PDFs
            new_pdf_restrictions = st.multiselect(
                "Select restricted PDFs",
                options=pdfs,
                default=current_pdf_restrictions,
                key=f"pdf_restrictions_{st.session_state.editing_role}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Changes"):
                    # Prevent rapid clicks (debounce)
                    current_time = time.time()
                    if current_time - st.session_state.last_save_time < 2:  # 2-second cooldown
                        st.warning("Please wait a moment before saving again.")
                    else:
                        st.session_state.last_save_time = current_time
                        # Deduplicate selections
                        new_db_restrictions = list(set(new_db_restrictions))
                        new_pdf_restrictions = list(set(new_pdf_restrictions))
                        
                        conn = sqlite3.connect("database/local_db.sqlite")
                        cursor = conn.cursor()
                        
                        # Clear existing restrictions for the role
                        cursor.execute("DELETE FROM db_access_control WHERE role = ?", (st.session_state.editing_role,))
                        cursor.execute("DELETE FROM pdf_access_control WHERE role = ?", (st.session_state.editing_role,))
                        
                        # Insert updated restrictions
                        for col in new_db_restrictions:
                            cursor.execute(
                                "INSERT OR IGNORE INTO db_access_control (role, restricted_column) VALUES (?, ?)",
                                (st.session_state.editing_role, col)
                            )
                        for pdf in new_pdf_restrictions:
                            cursor.execute(
                                "INSERT OR IGNORE INTO pdf_access_control (role, restricted_pdf) VALUES (?, ?)",
                                (st.session_state.editing_role, pdf)
                            )
                        
                        conn.commit()
                        conn.close()
                        st.success(f"Access controls updated for {st.session_state.editing_role}!")
                        st.session_state.editing_role = None
                        st.rerun()
            
            with col2:
                if st.button("Cancel"):
                    st.session_state.editing_role = None
                    st.rerun()
    
    with tab3:
        st.subheader("Performance Metrics")
        metrics = get_metrics()
        stats = calculate_stats(metrics)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        st.write(f"**Total Queries**: {stats['total_queries']}")
        st.write(f"**Average Response Time**: {stats['avg_response_time']:.2f} seconds")
        
        st.write("**By Query Type**")
        if stats["by_type"]:
            # Prepare data for charts
            query_types = list(stats["by_type"].keys())
            counts = [stats["by_type"][qt]["count"] for qt in query_types]
            avg_times = [stats["by_type"][qt]["avg_time"] for qt in query_types]
            
            # Display metrics
            for query_type, data in stats["by_type"].items():
                st.write(f"{query_type}: {data['count']} queries, Avg Time: {data['avg_time']:.2f} seconds")
            
            # Bar chart for query counts
            df_counts = pd.DataFrame({"Query Type": query_types, "Count": counts})
            st.subheader("Query Counts by Type")
            fig_counts = px.bar(df_counts, x="Query Type", y="Count", title="Number of Queries per Type")
            st.plotly_chart(fig_counts, use_container_width=True)
            
            # Line chart for average response times
            df_times = pd.DataFrame({"Query Type": query_types, "Average Response Time (s)": avg_times})
            st.subheader("Average Response Time by Type")
            fig_times = px.line(df_times, x="Query Type", y="Average Response Time (s)", title="Average Response Time per Type")
            st.plotly_chart(fig_times, use_container_width=True)
        else:
            st.write("No query type data available. Start using the chatbot to log queries and view metrics.")
        
        st.subheader("Query Log")
        
        with st.container():
            st.markdown(
                """
                <div style="max-height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                """,
                unsafe_allow_html=True
            )

            if metrics:
                log_data = []
                for m in metrics:
                    log_entry = {
                        "Timestamp": m["timestamp"],
                        "Query Type": m["query_type"],
                        "Question": m["question"],
                        "Response Time (s)": f"{m['response_time']:.2f}",
                        "Token Usage": m["token_usage"] or "N/A",
                        "Accuracy": m["accuracy"] or "N/A"
                    }
                    log_data.append(log_entry)
                    st.markdown(f"**{log_entry['Timestamp']}** | Type: {log_entry['Query Type']} | Question: {log_entry['Question']}")
                    st.markdown(f"Response Time: {log_entry['Response Time (s)']}s | Token Usage: {log_entry['Token Usage']} | Accuracy: {log_entry['Accuracy']}")
                    st.markdown("---")
            else:
                st.markdown("No queries logged yet.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Add download functionality
        if metrics:
            df_logs = pd.DataFrame(log_data)
            csv_logs = df_logs.to_csv(index=False)
            
            markdown_report = f"# Query Log Report - {current_time}\n\n"
            markdown_report += "## Log Entries\n\n"
            markdown_report += "\n".join(
                f"- **Timestamp:** {m['timestamp']}\n"
                f"- **Query Type:** {m['query_type']}\n"
                f"- **Question:** {m['question']}\n"
                f"- **Response Time:** {m['response_time']:.2f}s\n"
                f"- **Token Usage:** {m['token_usage']}\n"
                f"- **Accuracy:** {m['accuracy']}\n"
                for m in metrics
            )

            st.download_button(
                label="📥 Download CSV Report",
                data=csv_logs,
                file_name=f"query_log_{current_time}.csv",
                mime="text/csv"
            )
            
            st.download_button(
                label="📄 Download Markdown Report",
                data=markdown_report,
                file_name=f"query_log_{current_time}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()