import sqlite3
from typing import List, Dict, Optional
from datetime import datetime

def get_metrics() -> List[Dict]:
    try:
        conn = sqlite3.connect("database/local_db.sqlite")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, query_type, question, response_time, token_usage, accuracy
            FROM metrics
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "timestamp": row[0],
                "query_type": row[1],
                "question": row[2],
                "response_time": row[3],
                "token_usage": row[4],
                "accuracy": row[5]
            }
            for row in rows
        ]
    except Exception as e:
        print(f"[ERROR] Failed to retrieve metrics: {str(e)}")
        return []

def calculate_stats(metrics: List[Dict]) -> Dict:
    if not metrics:
        return {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "by_type": {}
        }
    
    total_queries = len(metrics)
    total_response_time = sum(i["response_time"] or 0 for i in metrics)
    avg_response_time = total_response_time / total_queries if total_queries > 0 else 0.0
    
    by_type = {}
    for i in metrics:
        query_type = i["query_type"] or "UNKNOWN"
        if query_type not in by_type:
            by_type[query_type] = {"count": 0, "response_times": []}
        by_type[query_type]["count"] += 1
        if i["response_time"] is not None:
            by_type[query_type]["response_times"].append(i["response_time"])
    
    # computing avg_time for each  type
    for query_type, data in by_type.items():
        count = data["count"]
        response_times = data["response_times"]
        data["avg_time"] = sum(response_times) / count if count > 0 and response_times else 0.0
    
    return {
        "total_queries": total_queries,
        "avg_response_time": avg_response_time,
        "by_type": by_type
    }

def log_metrics(query_type: str, question: str, response_time: float, token_usage: Optional[int] = None, accuracy: Optional[float] = None) -> None:
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect("database/local_db.sqlite")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO metrics (timestamp, query_type, question, response_time, token_usage, accuracy)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, query_type, question, response_time, token_usage, accuracy))
        conn.commit()
        conn.close()
        # print(f"[DEBUG] Logged metrics: {query_type}, {question}, {response_time}s")
    except Exception as e:
        print(f"[ERROR] Failed to log metrics: {str(e)}")