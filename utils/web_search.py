import json
import os
from typing import List, Dict, Optional
from langchain_community.utilities import GoogleSerperAPIWrapper  # Google Serper 
from utils.agent import call_bedrock_claude_3_haiku  
from dotenv import load_dotenv


load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("GOOGLE_SERPER_API_KEY")  

search_tool = GoogleSerperAPIWrapper()

def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        result = search_tool.results(query)
        # format result
        search_results = [
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", "")
            }
            for item in result.get("organic", [])
        ]
        return search_results
        # print(f"[DEBUG] Web search results for '{query}': {search_results}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse search results for '{query}': {str(e)}")
        return []
    except Exception as e:
        print(f"[ERROR] Web search failed: {str(e)}")
        return []

def process_search(search_results: List[Dict[str, str]], query: str) -> Optional[str]:
    if not search_results:
        return "No web search results available."

    search_text = "\n\n".join(
        f"Title: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}"
        for result in search_results
    )

    prompt = f"""Based on the following web search results, answer the query: '{query}'.

    Search Results:
    {search_text}

    Provide a concise and accurate response.
    """

    try:
        result = call_bedrock_claude_3_haiku(prompt=prompt, model_id="claude-3-haiku", max_tokens=500, temperature=0.2)
        return result
    except Exception as e:
        print(f"[ERROR] Claude 3 Haiku processing failed: {str(e)}")
        return f"Error processing with Claude: {str(e)}"

def web_search_and_process(query: str, max_results: int = 5) -> str:
    search_results = search_web(query, max_results)
    if not search_results:
        return "No results found on the web for your query. Try rephrasing it or check your internet connection."
    return process_search(search_results, query)


