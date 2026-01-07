# functions.py - User-Defined Functions (UDFs) for the Pixeltable Agent
# ---------------------------------------------------------------------------
# This file contains Python functions decorated with `@pxt.udf`.
# These UDFs define custom logic (e.g., API calls, data processing)
# that can be seamlessly integrated into Pixeltable workflows.
# Pixeltable automatically calls these functions when computing columns
# in tables or views (as defined in setup_pixeltable.py).
# ---------------------------------------------------------------------------

# Standard library imports
import os
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

# Third-party library imports
import requests
import yfinance as yf
from duckduckgo_search import DDGS

# Pixeltable library
import pixeltable as pxt
from pixeltable.func import Tools
from pixeltable import exprs

# Pixeltable UDFs (User-Defined Functions) extend the platform's capabilities.
# They allow you to wrap *any* Python code and use it within Pixeltable's
# declarative data processing and workflow engine.
# Pixeltable handles scheduling, execution, caching, and error handling.


# Tool UDF: Fetches latest news using NewsAPI.
# Registered as a tool for the LLM via `pxt.tools()` in setup_pixeltable.py.
@pxt.udf
def get_latest_news(topic: str) -> str:
    """Fetch latest news for a given topic using NewsAPI."""
    try:
        api_key = os.environ.get("NEWS_API_KEY")
        if not api_key:
            return "Error: NewsAPI key not found in environment variables."

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": topic,
            "apiKey": api_key,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 5,
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return f"Error: NewsAPI request failed ({response.status_code}): {response.text}"

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return f"No recent news found for '{topic}'."

        # Format multiple articles
        formatted_news = []
        for i, article in enumerate(articles[:3], 1):
            pub_date = datetime.fromisoformat(
                article["publishedAt"].replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
            formatted_news.append(
                f"{i}. [{pub_date}] {article['title']}\n   {article['description']}"
            )

        return "\n\n".join(formatted_news)

    except requests.Timeout:
        return "Error: NewsAPI request timed out."
    except requests.RequestException as e:
        return f"Error making NewsAPI request: {str(e)}"
    except Exception as e:
        return f"Unexpected error fetching news: {str(e)}."


# Tool UDF: Searches news using DuckDuckGo.
# Registered as a tool for the LLM via `pxt.tools()` in setup_pixeltable.py.
@pxt.udf
def search_news(keywords: str, max_results: int = 5) -> str:
    """Search news using DuckDuckGo and return results."""
    try:
        # DDGS requires entering the context manager explicitly
        with DDGS() as ddgs:
            results = list(
                ddgs.news(  # Convert iterator to list for processing
                    keywords=keywords,
                    region="wt-wt",
                    safesearch="off",
                    timelimit="m",  # Limit search to the last month
                    max_results=max_results,
                )
            )
            if not results:
                return "No news results found."

            # Format results for readability
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. Title: {r.get('title', 'N/A')}\n"
                    f"   Source: {r.get('source', 'N/A')}\n"
                    f"   Published: {r.get('date', 'N/A')}\n"
                    f"   URL: {r.get('url', 'N/A')}\n"
                    f"   Snippet: {r.get('body', 'N/A')}\n"
                )
            return "\n".join(formatted_results)
    except Exception as e:
        print(f"DuckDuckGo search failed: {str(e)}")
        return f"Search failed: {str(e)}."


# Tool UDF: Fetches financial data using yfinance.
# Integrates external Python libraries into the Pixeltable workflow.
# Registered as a tool for the LLM via `pxt.tools()` in setup_pixeltable.py.
@pxt.udf
def fetch_financial_data(ticker: str) -> str:
    """Fetch financial summary data for a given company ticker using yfinance."""
    try:
        if not ticker:
            return "Error: No ticker symbol provided."

        stock = yf.Ticker(ticker)

        # Get the info dictionary - this is the primary source now
        info = stock.info
        if (
            not info or info.get("quoteType") == "MUTUALFUND"
        ):  # Basic check if info exists and isn't a mutual fund (less relevant fields)
            # Attempt history for basic validation if info is sparse
            hist = stock.history(period="1d")
            if hist.empty:
                return f"Error: No data found for ticker '{ticker}'. It might be delisted or incorrect."
            else:  # Sometimes info is missing but history works, provide minimal info
                return f"Limited info for '{ticker}'. Previous Close: {hist['Close'].iloc[-1]:.2f} (if available)."

        # Select and format key fields from the info dictionary
        data_points = {
            "Company Name": info.get("shortName") or info.get("longName"),
            "Symbol": info.get("symbol"),
            "Exchange": info.get("exchange"),
            "Quote Type": info.get("quoteType"),
            "Currency": info.get("currency"),
            "Current Price": info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("bid"),
            "Previous Close": info.get("previousClose"),
            "Open": info.get("open"),
            "Day Low": info.get("dayLow"),
            "Day High": info.get("dayHigh"),
            "Volume": info.get("volume") or info.get("regularMarketVolume"),
            "Market Cap": info.get("marketCap"),
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "Dividend Yield": info.get("dividendYield"),
            "52 Week Low": info.get("fiftyTwoWeekLow"),
            "52 Week High": info.get("fiftyTwoWeekHigh"),
            "Avg Volume (10 day)": info.get("averageDailyVolume10Day"),
            # Add more fields if desired
        }

        formatted_data = [
            f"Financial Summary for {data_points.get('Company Name', ticker)} ({data_points.get('Symbol', ticker).upper()}) - {data_points.get('Quote Type', 'N/A')}"
        ]
        formatted_data.append("-" * 40)

        for key, value in data_points.items():
            if value is not None:  # Only show fields that have a value
                formatted_value = value
                # Format specific types for readability
                if key in [
                    "Current Price",
                    "Previous Close",
                    "Open",
                    "Day Low",
                    "Day High",
                    "52 Week Low",
                    "52 Week High",
                ] and isinstance(value, (int, float)):
                    formatted_value = (
                        f"{value:.2f} {data_points.get('Currency', '')}".strip()
                    )
                elif key in [
                    "Volume",
                    "Market Cap",
                    "Avg Volume (10 day)",
                ] and isinstance(value, (int, float)):
                    if value > 1_000_000_000:
                        formatted_value = f"{value / 1_000_000_000:.2f}B"
                    elif value > 1_000_000:
                        formatted_value = f"{value / 1_000_000:.2f}M"
                    elif value > 1_000:
                        formatted_value = f"{value / 1_000:.2f}K"
                    else:
                        formatted_value = f"{value:,}"
                elif key == "Dividend Yield" and isinstance(value, (int, float)):
                    formatted_value = f"{value * 100:.2f}%"
                elif (
                    key == "Trailing P/E"
                    or key == "Forward P/E") and isinstance(value, (int, float)
                ):
                    formatted_value = f"{value:.2f}"

                formatted_data.append(f"{key}: {formatted_value}")

        # Optionally, add a line about latest financials if easily available
        try:
            latest_financials = stock.financials.iloc[:, 0]
            revenue = latest_financials.get("Total Revenue")
            net_income = latest_financials.get("Net Income")
            if revenue is not None or net_income is not None:
                formatted_data.append("-" * 40)
                fin_date = latest_financials.name.strftime("%Y-%m-%d")
                if revenue:
                    formatted_data.append(
                        f"Latest Revenue ({fin_date}): ${revenue / 1e6:.2f}M"
                    )
                if net_income:
                    formatted_data.append(
                        f"Latest Net Income ({fin_date}): ${net_income / 1e6:.2f}M"
                    )
        except Exception:
            pass  # Ignore errors fetching/parsing financials for this summary

        return "\n".join(formatted_data)

    except Exception as e:
        traceback.print_exc()  # Log the full error for debugging
        return f"Error fetching financial data for {ticker}: {str(e)}."


# Context Assembly UDF: Combines various text-based search results and tool outputs.
# This function is called by a computed column in the `agents.tools` table
# to prepare the summarized context before the final LLM call.
# Demonstrates processing results from multiple Pixeltable search queries.
@pxt.udf
def assemble_multimodal_context(
    question: str,
    tool_outputs: Optional[List[Dict[str, Any]]],
    doc_context: Optional[List[Union[Dict[str, Any], str]]],
    memory_context: Optional[List[Dict[str, Any]]] = None,
    chat_memory_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Constructs a single text block summarizing various context types
    (documents, memory bank items, chat history search results, and generic tool outputs)
    relevant to the user's question.
    Video/Audio transcript results will appear in 'tool_outputs' if the LLM chose to call those tools.
    Does NOT include recent chat history or image/video frame details.
    """
    # --- Image Handling Note ---
    # Image/Video frame context is handled in `assemble_final_messages`
    # as it requires specific formatting for multimodal LLM input.

    # Format document context inline
    doc_context_str = "N/A"
    if doc_context:
        doc_items = []
        for item in doc_context:
            # Safely extract text and source filename
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            source = (
                item.get("source_doc", "Unknown Document")
                if isinstance(item, dict)
                else "Unknown Document"
            )
            source_name = os.path.basename(str(source))
            if text:
                doc_items.append(f"- [Source: {source_name}] {text}")
        if doc_items:
            doc_context_str = "\n".join(doc_items)

    # Format memory bank context
    memory_context_str = "N/A"
    if memory_context:
        memory_items = []
        for item in memory_context:
            # Safely extract details
            content = item.get("content", "")
            item_type = item.get("type", "unknown")
            language = item.get("language")
            sim = item.get("sim")
            context_query = item.get("context_query", "Unknown Query")
            # Use triple quotes for multi-line f-string clarity
            item_desc = f"""- [Memory Item | Type: {item_type}{f" ({language})" if language else ""} | Original Query: '{context_query}' {f"| Sim: {sim:.3f}" if sim is not None else ""}]
Content: {content[:100]}{"..." if len(content) > 100 else ""}"""
            memory_items.append(item_desc)
        if memory_items:
            memory_context_str = "\n".join(memory_items)

    # Format chat history search context
    chat_memory_context_str = "N/A"
    if chat_memory_context:
        chat_memory_items = []
        for item in chat_memory_context:
            content = item.get("content", "")
            role = item.get("role", "unknown")
            sim = item.get("sim")
            timestamp = item.get("timestamp")
            ts_str = (
                timestamp.strftime("%Y-%m-%d %H:%M") if timestamp else "Unknown Time"
            )
            item_desc = f"""- [Chat History | Role: {role} | Time: {ts_str} {f"| Sim: {sim:.3f}" if sim is not None else ""}]
Content: {content[:150]}{"..." if len(content) > 150 else ""}"""
            chat_memory_items.append(item_desc)
        if chat_memory_items:
            chat_memory_context_str = "\n".join(chat_memory_items)

    # Format tool outputs
    tool_outputs_str = str(tool_outputs) if tool_outputs else "N/A"

    # Construct the final summary text block
    # Ensure question is never None
    question_str = str(question) if question is not None else ""
    
    text_content = f"""
ORIGINAL QUESTION:
{question_str}

AVAILABLE CONTEXT:

[TOOL RESULTS]
{tool_outputs_str}

[DOCUMENT CONTEXT]
{doc_context_str}

[MEMORY BANK CONTEXT]
{memory_context_str}

[CHAT HISTORY SEARCH CONTEXT] (Older messages relevant to the query)
{chat_memory_context_str}
"""

    return text_content.strip()


# Final Message Assembly UDF: Creates the structured message list for the main LLM.
# This handles the specific format required by multimodal models (like Claude 3.5 Sonnet)
# incorporating text, images, and potentially video frames.
# It is called by a computed column in the `agents.tools` table.
@pxt.udf
def assemble_final_messages(
    history_context: Optional[List[Dict[str, Any]]],
    multimodal_context_text: str,
    image_context: Optional[List[Dict[str, Any]]] = None,  # Input image results
    video_frame_context: Optional[
        List[Dict[str, Any]]
    ] = None,  # Input video frame results
) -> List[Dict[str, Any]]:
    """
    Constructs the final list of messages for the LLM, incorporating:
    - Recent chat history (user/assistant turns).
    - The main text context summary (docs, memory, tool outputs, etc.).
    - Image context (base64 encoded images).
    - Video frame context (base64 encoded video frames).

    This structure is required for multimodal LLMs like Claude 3.

    Args:
        history_context: Recent chat messages.
        multimodal_context_text: The combined text context from `assemble_multimodal_context`.
        image_context: List of image search results (containing base64 data).
        video_frame_context: List of video frame search results (containing base64 data).

    Returns:
        A list of messages formatted for the LLM API.
    """
    messages = []

    # 1. Add recent chat history (if any) in chronological order
    # If images are present, we'll add history after checking for images
    has_images = image_context and len([item for item in image_context if isinstance(item, dict) and item.get("encoded_image")]) > 0
    
    if history_context:
        for item in reversed(history_context):
            role = item.get("role")
            content = item.get("content")
            # Filter out None values and ensure content is not None
            if role and content is not None:
                # If images are present, filter out tool-related messages to prevent tool usage
                if has_images:
                    # Skip messages that contain tool calls or tool-related content
                    content_str = str(content) if not isinstance(content, list) else str(content)
                    if "tool_calls" in content_str.lower() or "function_call" in content_str.lower():
                        continue  # Skip tool-related messages when analyzing images
                
                # If content is a list, filter out None values
                if isinstance(content, list):
                    content = [c for c in content if c is not None]
                    if not content:  # Skip if all content was None
                        continue
                messages.append({"role": role, "content": content})

    # 2. Prepare the content block for the final user message
    final_user_content = []

    # 2a. Add image blocks (if any)
    image_count = 0
    if image_context:
        for item in image_context:
            # Safely extract base64 encoded image data
            if isinstance(item, dict) and "encoded_image" in item:
                image_data = item["encoded_image"]
                # Skip if None
                if image_data is None:
                    continue
                # Ensure it's a string
                if isinstance(image_data, bytes):
                    image_data = image_data.decode("utf-8")
                elif not isinstance(image_data, str):
                    continue  # Skip invalid data
                
                # Skip if empty string
                if not image_data:
                    continue

                # Append in OpenAI's format for images
                # OpenAI expects data URIs directly
                if not image_data.startswith("data:"):
                    image_data = f"data:image/png;base64,{image_data}"
                final_user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }
                )
                image_count += 1
    
    # Log image count for debugging
    if image_count > 0:
        print(f"Added {image_count} image(s) to final user message")
    else:
        print("Warning: No images were added to final user message despite image_context being provided")

    # 2b. Add video frame blocks (if any) - NOTE: Currently illustrative, LLM support varies
    if video_frame_context:
        for item in video_frame_context:
            # Safely extract base64 encoded video frame data
            if isinstance(item, dict) and "encoded_frame" in item:
                video_frame_data = item["encoded_frame"]
                # Skip if None
                if video_frame_data is None:
                    continue
                if isinstance(video_frame_data, bytes):
                    video_frame_data = video_frame_data.decode("utf-8")
                elif not isinstance(video_frame_data, str):
                    continue  # Skip invalid data
                
                # Skip if empty string
                if not video_frame_data:
                    continue

                # Append in OpenAI's format for video frames (treated as images)
                if not video_frame_data.startswith("data:"):
                    video_frame_data = f"data:image/png;base64,{video_frame_data}"
                final_user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": video_frame_data
                        }
                    }
                )

    # 2c. Add the main text context block (ensure it's never None)
    # If images are present, prioritize image analysis and completely remove tool-related context
    if image_count > 0:
        # When images are present, create a focused prompt that emphasizes image analysis
        # Extract just the question from the context, removing all tool-related content
        question_str = ""
        if multimodal_context_text:
            context_str = str(multimodal_context_text)
            # Extract question if present
            if "ORIGINAL QUESTION:" in context_str:
                parts = context_str.split("ORIGINAL QUESTION:")
                if len(parts) > 1:
                    question_str = parts[1].split("\n")[0].strip()
            else:
                # Try to extract a simple question from the context
                lines = context_str.split("\n")
                for line in lines:
                    if line.strip() and not line.strip().startswith("[") and "tool" not in line.lower():
                        question_str = line.strip()[:200]  # Limit length
                        break
        
        if not question_str:
            question_str = "Please analyze the image(s) provided."
        
        # Create a completely clean, image-focused prompt with NO tool references
        text_content = f"""You are analyzing {image_count} image(s) provided above.

USER REQUEST: {question_str}

YOUR TASK:
Analyze the image(s) using your vision capabilities and describe what you see in detail.

IMPORTANT RULES:
- You MUST analyze the image(s) directly - they are provided above
- You MUST respond with text only - describe what you see
- You MUST NOT use any tools, functions, or external APIs
- You MUST NOT attempt function calls
- Simply look at the image(s) and describe them"""
    else:
        # No images, use the full context as before
        text_content = str(multimodal_context_text) if multimodal_context_text is not None else ""
    
    final_user_content.append(
        {
            "type": "text",
            "text": text_content,
        }
    )

    # 3. Append the complete user message (potentially multimodal)
    # Always add the message, even if only text (ensures we have at least one content item)
    messages.append({"role": "user", "content": final_user_content})

    return messages


# Follow-up Prompt Assembly UDF: Creates the input prompt for the follow-up LLM.
# Encapsulates the prompt template structure, making the workflow definition
# in setup_pixeltable.py cleaner and focusing it on data flow.
@pxt.udf
def assemble_follow_up_prompt(original_prompt: str, answer_text: str) -> str:
    """Constructs the formatted prompt string for the follow-up question LLM.

    This function encapsulates the prompt template to make it reusable and
    easier to see the input being sent to the LLM in the Pixeltable trace.
    Includes a few-shot example to guide the model.
    """
    # Updated template with clearer instructions and an example
    follow_up_system_prompt_template = """You are an expert assistant tasked with generating **exactly 3** relevant and concise follow-up questions based on an original user query and the provided answer. Focus *only* on the content provided.

**Instructions:**
1.  Read the <ORIGINAL_PROMPT_START> and <ANSWER_TEXT_START> sections carefully.
2.  Generate 3 distinct questions that logically follow from the information presented.
3.  The questions should encourage deeper exploration of the topic discussed.
4.  **Output ONLY the 3 questions**, one per line. Do NOT include numbering, bullet points, or any other text.

**Example:**

<ORIGINAL_PROMPT_START>
What are the main benefits of using Pixeltable for AI workflows?
</ORIGINAL_PROMPT_END>

<ANSWER_TEXT_START>
Pixeltable simplifies AI workflows by providing automated data orchestration, native multimodal support (text, images, video, audio), a declarative interface, and integrations with LLMs and ML models. It handles complex tasks like data versioning, incremental computation, and vector indexing automatically.
</ANSWER_TEXT_END>

How does Pixeltable handle data versioning specifically?
Can you elaborate on the declarative interface of Pixeltable?
What specific LLMs and ML models does Pixeltable integrate with?

**Now, generate questions for the following input:**

<ORIGINAL_PROMPT_START>
{original_prompt}
</ORIGINAL_PROMPT_END>

<ANSWER_TEXT_START>
{answer_text}
</ANSWER_TEXT_END>
"""
    return follow_up_system_prompt_template.format(
        original_prompt=original_prompt, answer_text=answer_text
    )


# Helper function to recursively sanitize nested structures (removes None values)
def _recursive_sanitize(obj: Any) -> Any:
    """Recursively removes None values from nested dictionaries and lists.
    Preserves empty strings and empty lists/dicts as they are valid values."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            sanitized_value = _recursive_sanitize(value)
            if sanitized_value is not None:
                sanitized[key] = sanitized_value
        # Return empty dict if all values were None (but this shouldn't happen in practice)
        return sanitized
    elif isinstance(obj, list):
        sanitized = []
        for item in obj:
            sanitized_item = _recursive_sanitize(item)
            if sanitized_item is not None:
                sanitized.append(sanitized_item)
        return sanitized
    else:
        # For primitive types (str, int, float, bool), return as-is
        return obj


# Helper function to sanitize messages for OpenAI (removes None values)
@pxt.udf
def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitizes messages by removing None values to prevent token counting errors."""
    sanitized = []
    for msg in messages:
        if msg is None:
            continue
        # Recursively sanitize the message to remove all None values
        sanitized_msg = _recursive_sanitize(msg)
        if sanitized_msg is None:
            continue
        # Ensure it's a dict and has required fields
        if not isinstance(sanitized_msg, dict):
            continue
        # Ensure role and content exist
        if "role" not in sanitized_msg:
            continue
        # If content is missing or None after sanitization, set to empty string
        if "content" not in sanitized_msg or sanitized_msg["content"] is None:
            sanitized_msg["content"] = ""
        # If content is a list and empty after sanitization, set to empty string
        elif isinstance(sanitized_msg["content"], list) and len(sanitized_msg["content"]) == 0:
            sanitized_msg["content"] = ""
        sanitized.append(sanitized_msg)
    return sanitized if sanitized else [{"role": "user", "content": ""}]  # Return at least one message


# Helper function to sanitize model_kwargs for OpenAI (removes None values)
@pxt.udf
def sanitize_model_kwargs(
    max_tokens: Optional[int],
    stop_sequences: Optional[pxt.Json],  # Use Json type for flexibility
    temperature: Optional[float],
    top_p: Optional[float],
) -> pxt.Json:
    """Sanitizes model kwargs by removing None values and ensuring proper types."""
    kwargs = {}
    
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    # OpenAI accepts stop as a string or list of strings, but not None
    if stop_sequences is not None:
        # Ensure it's a list of strings
        if isinstance(stop_sequences, str):
            kwargs["stop"] = stop_sequences
        elif isinstance(stop_sequences, list):
            # Filter out None values and ensure all are strings
            stop_list = [str(s) for s in stop_sequences if s is not None]
            if stop_list:
                kwargs["stop"] = stop_list if len(stop_list) > 1 else stop_list[0]
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    if top_p is not None:
        kwargs["top_p"] = top_p
    
    return kwargs


# Helper function to sanitize model_kwargs for final response
# Note: We do NOT set tool_choice here because OpenAI API doesn't allow tool_choice
# when tools=None. Simply omitting tools and tool_choice is sufficient to disable tools.
@pxt.udf
def sanitize_model_kwargs_final_response(
    max_tokens: Optional[int],
    stop_sequences: Optional[pxt.Json],
    temperature: Optional[float],
    top_p: Optional[float],
) -> pxt.Json:
    """Sanitizes model kwargs for final response. Tools are disabled by not passing them."""
    kwargs = {}
    
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    if stop_sequences is not None:
        if isinstance(stop_sequences, str):
            kwargs["stop"] = stop_sequences
        elif isinstance(stop_sequences, list):
            stop_list = [str(s) for s in stop_sequences if s is not None]
            if stop_list:
                kwargs["stop"] = stop_list if len(stop_list) > 1 else stop_list[0]
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    if top_p is not None:
        kwargs["top_p"] = top_p
    
    # Do NOT set tool_choice - OpenAI API doesn't allow it when tools=None
    # Simply omitting tools and tool_choice is sufficient to disable tools
    
    return kwargs


# Helper function to sanitize model_kwargs with explicit tool_choice
@pxt.udf
def sanitize_model_kwargs_with_tool_choice(
    max_tokens: Optional[int],
    stop_sequences: Optional[pxt.Json],
    temperature: Optional[float],
    top_p: Optional[float],
    tool_choice: Optional[str] = None,
) -> pxt.Json:
    """Sanitizes model kwargs and optionally sets tool_choice to explicitly disable tools."""
    kwargs = {}
    
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    # OpenAI accepts stop as a string or list of strings, but not None
    if stop_sequences is not None:
        # Ensure it's a list of strings
        if isinstance(stop_sequences, str):
            kwargs["stop"] = stop_sequences
        elif isinstance(stop_sequences, list):
            # Filter out None values and ensure all are strings
            stop_list = [str(s) for s in stop_sequences if s is not None]
            if stop_list:
                kwargs["stop"] = stop_list if len(stop_list) > 1 else stop_list[0]
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    if top_p is not None:
        kwargs["top_p"] = top_p
    
    # Explicitly set tool_choice if provided (e.g., 'none' to disable tools)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    
    return kwargs


# Helper function to combine system message with other messages for OpenAI
@pxt.udf
def combine_messages_with_system(system_prompt: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combines a system prompt with a list of messages for OpenAI chat completions."""
    # Filter out None values and ensure system prompt is not None
    if system_prompt is None:
        system_prompt = ""
    
    result = [{"role": "system", "content": system_prompt}]
    if messages:
        # Filter out any None messages and ensure content is not None
        filtered_messages = []
        for msg in messages:
            if msg is None:
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role and content is not None:
                # If content is a list, filter out None values
                if isinstance(content, list):
                    content = [c for c in content if c is not None]
                    if not content:  # Skip if all content was None
                        continue
                filtered_messages.append({"role": role, "content": content})
        result.extend(filtered_messages)
    
    # Ensure we always have at least one user message
    # If no messages were added, add a minimal user message to prevent API errors
    if len(result) == 1:  # Only system message
        print("Warning: No user messages found after filtering. Adding minimal user message.")
        result.append({"role": "user", "content": "Please provide a response."})
    
    return result


# OpenAI Tool Invocation Helper
# Converts OpenAI's response format to Pixeltable's tool invocation format
@pxt.udf
def _openai_response_to_pxt_tool_calls(response: dict) -> dict | None:
    """Converts an OpenAI response dict to Pixeltable tool invocation format."""
    if not response or 'choices' not in response or len(response['choices']) == 0:
        return None
    
    message = response['choices'][0].get('message', {})
    tool_calls = message.get('tool_calls', [])
    
    if len(tool_calls) == 0:
        return None
    
    pxt_tool_calls: dict[str, list[dict[str, Any]]] = {}
    for tool_call in tool_calls:
        function_info = tool_call.get('function', {})
        tool_name = function_info.get('name')
        if not tool_name:
            continue
        
        if tool_name not in pxt_tool_calls:
            pxt_tool_calls[tool_name] = []
        
        # Parse the arguments JSON string
        args_str = function_info.get('arguments', '{}')
        try:
            import json
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}
        
        pxt_tool_calls[tool_name].append({'args': args})
    
    return pxt_tool_calls if pxt_tool_calls else None


def invoke_openai_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an OpenAI response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


# Helper function to combine semantic search results with recent images
@pxt.udf
def combine_image_context(
    semantic_results: Optional[List[Dict[str, Any]]],
    recent_images: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Combines semantic search results with recent images.
    Prioritizes semantic matches but includes recent images if semantic results are empty.
    """
    combined = []
    
    # Add semantic search results first (they have higher priority)
    if semantic_results:
        for item in semantic_results:
            if isinstance(item, dict) and item.get("encoded_image"):
                combined.append(item)
    
    # If we have semantic results, we might still want to include recent images
    # But let's only add recent images if we have fewer than 3 semantic results
    if recent_images and len(combined) < 3:
        # Track which images we've already added (by checking if encoded_image matches)
        added_images = {item.get("encoded_image") for item in combined if item.get("encoded_image")}
        
        for item in recent_images:
            if isinstance(item, dict) and item.get("encoded_image"):
                # Only add if not already in combined results
                if item.get("encoded_image") not in added_images:
                    combined.append(item)
                    added_images.add(item.get("encoded_image"))
                    if len(combined) >= 5:  # Limit total to 5 images
                        break
    
    # If no semantic results, use recent images
    if not combined and recent_images:
        for item in recent_images:
            if isinstance(item, dict) and item.get("encoded_image"):
                combined.append(item)
                if len(combined) >= 3:  # Limit to 3 recent images if no semantic matches
                    break
    
    return combined if combined else None


# Helper function to safely extract answer from OpenAI response
@pxt.udf
def extract_openai_answer(response: dict) -> str:
    """Safely extracts the answer text from an OpenAI chat completion response."""
    if not response:
        return "Error: No response from OpenAI (response is None or empty)."
    
    # Check for API errors in the response
    if 'error' in response:
        error_info = response['error']
        error_msg = error_info.get('message', 'Unknown error') if isinstance(error_info, dict) else str(error_info)
        error_type = error_info.get('type', 'Unknown') if isinstance(error_info, dict) else 'Unknown'
        return f"Error from OpenAI API ({error_type}): {error_msg}"
    
    if 'choices' not in response or len(response['choices']) == 0:
        # Log the full response for debugging (but don't include it in the user-facing error)
        print(f"OpenAI response missing choices. Response keys: {list(response.keys())}")
        return "Error: No choices in OpenAI response. The API may have encountered an issue."
    
    choice = response['choices'][0]
    message = choice.get('message', {})
    
    # Check finish_reason to understand why content might be empty
    finish_reason = choice.get('finish_reason')
    
    # Get content first to check if it exists
    content = message.get('content')
    
    # Handle tool_calls finish_reason - this shouldn't happen in final response
    # but we need to handle it gracefully
    if finish_reason == 'tool_calls':
        tool_calls = message.get('tool_calls', [])
        print(f"OpenAI finish_reason: tool_calls. Tool calls count: {len(tool_calls)}")
        if tool_calls:
            tool_names = [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
            print(f"OpenAI attempted to call tools: {tool_names}")
        
        # Check if there's any content before the tool call
        if content is not None and str(content).strip():
            # Return the partial content if it exists
            return str(content).strip()
        else:
            # No content available - this means the model tried to use tools when it should analyze images directly
            # Provide a helpful error message that suggests the issue
            return "Error: The model attempted to use tools instead of analyzing the image directly. This may indicate that images were not properly passed to the model, or the model configuration needs adjustment. Please ensure images are uploaded and try again."
    
    if finish_reason and finish_reason != 'stop':
        reason_msg = {
            'length': 'Response was cut off due to token limit.',
            'content_filter': 'Response was filtered by content policy.',
            'function_call': 'Model requested a function call.',
        }.get(finish_reason, f'Response finished with reason: {finish_reason}')
        print(f"OpenAI finish_reason: {finish_reason}")
    
    if content is None:
        # Provide more context about why content is None
        if finish_reason == 'content_filter':
            return "Error: Response was filtered by OpenAI's content policy. Please try rephrasing your query."
        elif finish_reason == 'length':
            return "Error: Response was truncated due to token limit. The answer may be incomplete."
        else:
            print(f"OpenAI response has None content. Finish reason: {finish_reason}, Message keys: {list(message.keys())}")
            return f"Error: Empty response from OpenAI (finish_reason: {finish_reason or 'unknown'})."
    
    content_str = str(content) if content else ""
    if not content_str.strip():
        return "Error: Empty content in OpenAI response."
    
    return content_str
