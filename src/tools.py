"""
Tool definitions for the experiment.

This module defines the 10 base tools used in the experiment, plus utilities
for generating additional tools for scaling experiments.
"""

# Base tool set - 10 tools with distinct purposes
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "location": {"type": "string", "description": "City name or coordinates"},
        },
        "required": ["location"],
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "path": {"type": "string", "description": "File path to read"},
        },
        "required": ["path"],
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    },
    {
        "name": "run_code",
        "description": "Execute Python code",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
        },
        "required": ["to", "subject", "body"],
    },
    {
        "name": "get_calendar",
        "description": "Get calendar events for a date",
        "parameters": {
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
        },
        "required": ["date"],
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder for a specific time",
        "parameters": {
            "message": {"type": "string", "description": "Reminder message"},
            "time": {"type": "string", "description": "Time for reminder (ISO 8601)"},
        },
        "required": ["message", "time"],
    },
    {
        "name": "search_database",
        "description": "Query a database",
        "parameters": {
            "query": {"type": "string", "description": "SQL query to execute"},
        },
        "required": ["query"],
    },
    {
        "name": "translate_text",
        "description": "Translate text between languages",
        "parameters": {
            "text": {"type": "string", "description": "Text to translate"},
            "source_lang": {"type": "string", "description": "Source language code"},
            "target_lang": {"type": "string", "description": "Target language code"},
        },
        "required": ["text", "target_lang"],
    },
]

# Additional tools for scaling experiments (20+ tools)
EXTRA_TOOLS = [
    {
        "name": "convert_currency",
        "description": "Convert between currencies",
        "parameters": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "Source currency code"},
            "to_currency": {"type": "string", "description": "Target currency code"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
    {
        "name": "get_stock_price",
        "description": "Get current stock price",
        "parameters": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
        },
        "required": ["symbol"],
    },
    {
        "name": "create_image",
        "description": "Generate an image from a text description",
        "parameters": {
            "prompt": {"type": "string", "description": "Image description"},
        },
        "required": ["prompt"],
    },
    {
        "name": "summarize_text",
        "description": "Summarize a long text into key points",
        "parameters": {
            "text": {"type": "string", "description": "Text to summarize"},
        },
        "required": ["text"],
    },
    {
        "name": "get_directions",
        "description": "Get directions between two locations",
        "parameters": {
            "origin": {"type": "string", "description": "Starting location"},
            "destination": {"type": "string", "description": "Ending location"},
        },
        "required": ["origin", "destination"],
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a specific time",
        "parameters": {
            "time": {"type": "string", "description": "Alarm time"},
            "label": {"type": "string", "description": "Alarm label"},
        },
        "required": ["time"],
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {
            "query": {"type": "string", "description": "Song or playlist name"},
        },
        "required": ["query"],
    },
    {
        "name": "order_food",
        "description": "Order food from a restaurant",
        "parameters": {
            "restaurant": {"type": "string", "description": "Restaurant name"},
            "items": {"type": "array", "description": "List of items to order"},
        },
        "required": ["restaurant", "items"],
    },
    {
        "name": "book_flight",
        "description": "Book a flight ticket",
        "parameters": {
            "origin": {"type": "string", "description": "Departure city"},
            "destination": {"type": "string", "description": "Arrival city"},
            "date": {"type": "string", "description": "Travel date"},
        },
        "required": ["origin", "destination", "date"],
    },
    {
        "name": "reserve_hotel",
        "description": "Reserve a hotel room",
        "parameters": {
            "location": {"type": "string", "description": "Hotel location"},
            "checkin": {"type": "string", "description": "Check-in date"},
            "checkout": {"type": "string", "description": "Check-out date"},
        },
        "required": ["location", "checkin", "checkout"],
    },
    {
        "name": "calculate_tip",
        "description": "Calculate tip for a bill",
        "parameters": {
            "amount": {"type": "number", "description": "Bill amount"},
            "percentage": {"type": "number", "description": "Tip percentage"},
        },
        "required": ["amount"],
    },
    {
        "name": "convert_units",
        "description": "Convert between measurement units",
        "parameters": {
            "value": {"type": "number", "description": "Value to convert"},
            "from_unit": {"type": "string", "description": "Source unit"},
            "to_unit": {"type": "string", "description": "Target unit"},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    {
        "name": "find_restaurant",
        "description": "Find restaurants nearby",
        "parameters": {
            "location": {"type": "string", "description": "Search location"},
            "cuisine": {"type": "string", "description": "Type of cuisine"},
        },
        "required": ["location"],
    },
    {
        "name": "get_news",
        "description": "Get latest news headlines",
        "parameters": {
            "topic": {"type": "string", "description": "News topic"},
        },
        "required": [],
    },
    {
        "name": "check_spelling",
        "description": "Check spelling of text",
        "parameters": {
            "text": {"type": "string", "description": "Text to check"},
        },
        "required": ["text"],
    },
    {
        "name": "define_word",
        "description": "Get definition of a word",
        "parameters": {
            "word": {"type": "string", "description": "Word to define"},
        },
        "required": ["word"],
    },
    {
        "name": "find_synonym",
        "description": "Find synonyms for a word",
        "parameters": {
            "word": {"type": "string", "description": "Word to find synonyms for"},
        },
        "required": ["word"],
    },
    {
        "name": "generate_password",
        "description": "Generate a secure password",
        "parameters": {
            "length": {"type": "integer", "description": "Password length"},
        },
        "required": [],
    },
    {
        "name": "compress_file",
        "description": "Compress a file or folder",
        "parameters": {
            "path": {"type": "string", "description": "File path to compress"},
        },
        "required": ["path"],
    },
    {
        "name": "extract_archive",
        "description": "Extract a compressed archive",
        "parameters": {
            "path": {"type": "string", "description": "Archive path to extract"},
        },
        "required": ["path"],
    },
    {
        "name": "resize_image",
        "description": "Resize an image to specified dimensions",
        "parameters": {
            "path": {"type": "string", "description": "Image path"},
            "width": {"type": "integer", "description": "Target width"},
            "height": {"type": "integer", "description": "Target height"},
        },
        "required": ["path", "width", "height"],
    },
    {
        "name": "convert_video",
        "description": "Convert video to different format",
        "parameters": {
            "path": {"type": "string", "description": "Video path"},
            "format": {"type": "string", "description": "Target format"},
        },
        "required": ["path", "format"],
    },
    {
        "name": "send_sms",
        "description": "Send an SMS text message",
        "parameters": {
            "phone": {"type": "string", "description": "Phone number"},
            "message": {"type": "string", "description": "Message text"},
        },
        "required": ["phone", "message"],
    },
    {
        "name": "make_call",
        "description": "Make a phone call",
        "parameters": {
            "phone": {"type": "string", "description": "Phone number to call"},
        },
        "required": ["phone"],
    },
    {
        "name": "take_screenshot",
        "description": "Take a screenshot of the screen",
        "parameters": {
            "region": {"type": "string", "description": "Screen region to capture"},
        },
        "required": [],
    },
    {
        "name": "record_audio",
        "description": "Record audio from microphone",
        "parameters": {
            "duration": {"type": "integer", "description": "Recording duration in seconds"},
        },
        "required": ["duration"],
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribe audio to text",
        "parameters": {
            "path": {"type": "string", "description": "Audio file path"},
        },
        "required": ["path"],
    },
    {
        "name": "text_to_speech",
        "description": "Convert text to spoken audio",
        "parameters": {
            "text": {"type": "string", "description": "Text to speak"},
        },
        "required": ["text"],
    },
    {
        "name": "get_contacts",
        "description": "Get contacts from address book",
        "parameters": {
            "search": {"type": "string", "description": "Search query"},
        },
        "required": [],
    },
    {
        "name": "add_contact",
        "description": "Add a new contact",
        "parameters": {
            "name": {"type": "string", "description": "Contact name"},
            "phone": {"type": "string", "description": "Phone number"},
            "email": {"type": "string", "description": "Email address"},
        },
        "required": ["name"],
    },
    {
        "name": "get_location",
        "description": "Get current GPS location",
        "parameters": {},
        "required": [],
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {
            "duration": {"type": "string", "description": "Timer duration"},
        },
        "required": ["duration"],
    },
    {
        "name": "calculate_math",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression to evaluate"},
        },
        "required": ["expression"],
    },
    {
        "name": "get_time",
        "description": "Get current time in a timezone",
        "parameters": {
            "timezone": {"type": "string", "description": "Timezone name"},
        },
        "required": [],
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting with attendees",
        "parameters": {
            "title": {"type": "string", "description": "Meeting title"},
            "time": {"type": "string", "description": "Meeting time"},
            "attendees": {"type": "array", "description": "List of attendees"},
        },
        "required": ["title", "time"],
    },
    {
        "name": "create_note",
        "description": "Create a new note",
        "parameters": {
            "title": {"type": "string", "description": "Note title"},
            "content": {"type": "string", "description": "Note content"},
        },
        "required": ["content"],
    },
    {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "path": {"type": "string", "description": "Directory path"},
        },
        "required": ["path"],
    },
    {
        "name": "delete_file",
        "description": "Delete a file",
        "parameters": {
            "path": {"type": "string", "description": "File path to delete"},
        },
        "required": ["path"],
    },
    {
        "name": "move_file",
        "description": "Move a file to new location",
        "parameters": {
            "source": {"type": "string", "description": "Source path"},
            "destination": {"type": "string", "description": "Destination path"},
        },
        "required": ["source", "destination"],
    },
    {
        "name": "copy_file",
        "description": "Copy a file to new location",
        "parameters": {
            "source": {"type": "string", "description": "Source path"},
            "destination": {"type": "string", "description": "Destination path"},
        },
        "required": ["source", "destination"],
    },
]


def tools_to_qwen_format(tools: list[dict]) -> list[dict]:
    """
    Convert tool definitions to Qwen3's expected format.

    Args:
        tools: List of tool definitions with name, description, parameters, required

    Returns:
        List of tools in Qwen3's OpenAI-compatible format
    """
    qwen_tools = []
    for tool in tools:
        qwen_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool.get("parameters", {}),
                    "required": tool.get("required", []),
                },
            },
        })
    return qwen_tools


def get_tools_subset(n: int) -> list[dict]:
    """
    Get a subset of tools for scaling experiments.

    Args:
        n: Number of tools to return (2-50)

    Returns:
        List of n tools
    """
    if n <= 10:
        return TOOLS[:n]
    else:
        # Combine base tools with extra tools
        all_tools = TOOLS + EXTRA_TOOLS
        return all_tools[:n]


def get_tool_names(tools: list[dict] | None = None) -> list[str]:
    """Get list of tool names."""
    if tools is None:
        tools = TOOLS
    return [t["name"] for t in tools]
