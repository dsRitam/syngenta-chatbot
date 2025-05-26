import sqlite3

def init_db():
    conn = sqlite3.connect("database/local_db.sqlite")
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            country TEXT NOT NULL
        )
    """)
    
    # Create db_access_control table 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS db_access_control (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            restricted_column TEXT NOT NULL,
            UNIQUE(role, restricted_column)
        )
    """)
    
    # Create pdf_access_control table 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_access_control (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            restricted_pdf TEXT NOT NULL,
            UNIQUE(role, restricted_pdf)
        )
    """)
    
    # Create metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query_type TEXT NOT NULL,
            question TEXT NOT NULL,
            response_time REAL NOT NULL,
            token_usage INTEGER,
            accuracy REAL
        )
    """)
    
    # Insert admin credentials on startup
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'admin@email.com'")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, email, password, role, country) VALUES (?, ?, ?, ?, ?)",
            ("Admin", "admin@email.com", "admin@123", "admin", "Global")
        )
    
    conn.commit()
    conn.close()