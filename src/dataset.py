"""
Dataset generation and management.

This module handles creating the evaluation dataset of prompts
that unambiguously require specific tools.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict

from src.tools import TOOLS, get_tool_names


@dataclass
class EvalPrompt:
    """A single evaluation prompt."""
    prompt: str
    expected_tool: str
    category: str = ""  # Optional category for analysis


# Seed prompts for each tool (5 examples each as templates)
SEED_PROMPTS = {
    "get_weather": [
        "What's the weather in Tokyo?",
        "Is it going to rain in London tomorrow?",
        "Tell me the current temperature in New York",
        "What's the forecast for Paris this weekend?",
        "How's the weather looking in Sydney?",
    ],
    "search_web": [
        "Find recent news about SpaceX launches",
        "Search for the best restaurants in San Francisco",
        "Look up information about the Python programming language",
        "Find articles about climate change",
        "Search for reviews of the latest iPhone",
    ],
    "read_file": [
        "Show me the contents of config.yaml",
        "Read the README.md file",
        "What's in the settings.json file?",
        "Display the contents of main.py",
        "Open and show me data.csv",
    ],
    "write_file": [
        "Save this text to notes.txt: Hello world",
        "Create a file called output.json with the results",
        "Write the following to config.yaml: debug: true",
        "Save my notes to diary.txt",
        "Create a new file report.md with the summary",
    ],
    "run_code": [
        "Execute this Python code: print('Hello')",
        "Run this script: for i in range(10): print(i)",
        "Calculate 2 + 2 using Python",
        "Execute: import math; print(math.pi)",
        "Run this code to check the result",
    ],
    "send_email": [
        "Send an email to john@example.com about the meeting",
        "Email my boss that I'll be late today",
        "Send a message to support@company.com asking about my order",
        "Email the team about the project update",
        "Send an email to hr@company.com with my vacation request",
    ],
    "get_calendar": [
        "What meetings do I have today?",
        "Show my calendar for next Monday",
        "What's on my schedule for December 15th?",
        "List my appointments for this week",
        "Do I have any events on Friday?",
    ],
    "create_reminder": [
        "Remind me to call mom at 5pm",
        "Set a reminder for the meeting tomorrow at 10am",
        "Create a reminder to buy groceries this evening",
        "Remind me to submit the report by Friday",
        "Set a reminder for my doctor's appointment next week",
    ],
    "search_database": [
        "Query the database for all users created last month",
        "Find all orders with status 'pending'",
        "Search the database for products under $50",
        "Get all customers from California",
        "Query for employees in the engineering department",
    ],
    "translate_text": [
        "Translate 'Hello, how are you?' to Spanish",
        "Convert this text to French: Good morning",
        "Translate 'Thank you very much' to Japanese",
        "How do you say 'Where is the train station?' in German?",
        "Translate this paragraph to Chinese",
    ],
}


def generate_prompt_variations(
    seed_prompt: str,
    tool_name: str,
    n_variations: int = 10,
) -> list[str]:
    """
    Generate variations of a seed prompt.

    This is a simple rule-based generator. For production,
    you'd want to use an LLM to generate diverse variations.

    Args:
        seed_prompt: The template prompt
        tool_name: The expected tool
        n_variations: Number of variations to generate

    Returns:
        List of prompt variations
    """
    variations = [seed_prompt]

    # Simple variations: add prefixes/suffixes
    prefixes = [
        "Can you ",
        "Please ",
        "I need you to ",
        "Could you ",
        "I want to ",
        "Help me ",
        "",
    ]

    suffixes = [
        "",
        " please",
        " for me",
        " right now",
        " as soon as possible",
    ]

    for prefix in prefixes:
        for suffix in suffixes:
            if len(variations) >= n_variations:
                break
            var = prefix + seed_prompt.lower() + suffix
            var = var[0].upper() + var[1:]  # Capitalize first letter
            if var not in variations:
                variations.append(var)

    return variations[:n_variations]


def generate_dataset(
    prompts_per_tool: int = 50,
    tools: list[dict] | None = None,
) -> list[EvalPrompt]:
    """
    Generate the full evaluation dataset.

    Args:
        prompts_per_tool: Number of prompts per tool
        tools: Tool definitions (defaults to TOOLS)

    Returns:
        List of EvalPrompt objects
    """
    if tools is None:
        tools = TOOLS

    tool_names = get_tool_names(tools)
    dataset = []

    for tool_name in tool_names:
        seeds = SEED_PROMPTS.get(tool_name, [])

        if not seeds:
            print(f"Warning: No seed prompts for {tool_name}")
            continue

        # Generate variations from each seed
        variations_per_seed = prompts_per_tool // len(seeds) + 1
        tool_prompts = []

        for seed in seeds:
            variations = generate_prompt_variations(
                seed, tool_name, n_variations=variations_per_seed
            )
            tool_prompts.extend(variations)

        # Take exactly prompts_per_tool, shuffled
        random.shuffle(tool_prompts)
        tool_prompts = tool_prompts[:prompts_per_tool]

        for prompt in tool_prompts:
            dataset.append(EvalPrompt(
                prompt=prompt,
                expected_tool=tool_name,
            ))

    return dataset


def save_dataset(dataset: list[EvalPrompt], path: str | Path):
    """Save dataset to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump([asdict(p) for p in dataset], f, indent=2)

    print(f"Saved {len(dataset)} prompts to {path}")


def load_dataset(path: str | Path) -> list[EvalPrompt]:
    """Load dataset from JSON file."""
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    return [EvalPrompt(**d) for d in data]


def get_dataset_stats(dataset: list[EvalPrompt]) -> dict:
    """Get statistics about the dataset."""
    tool_counts = {}
    for prompt in dataset:
        tool = prompt.expected_tool
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    return {
        "total_prompts": len(dataset),
        "tools": len(tool_counts),
        "prompts_per_tool": tool_counts,
        "min_per_tool": min(tool_counts.values()) if tool_counts else 0,
        "max_per_tool": max(tool_counts.values()) if tool_counts else 0,
    }


# Pre-defined diverse prompts for higher quality evaluation
# These are manually crafted to be unambiguous
CURATED_PROMPTS = {
    "get_weather": [
        "What's the weather in Tokyo?",
        "Is it going to rain in London tomorrow?",
        "Tell me the current temperature in New York",
        "What's the forecast for Paris this weekend?",
        "How's the weather looking in Sydney?",
        "Will I need an umbrella in Seattle today?",
        "What's the humidity level in Miami right now?",
        "Is it snowing in Denver?",
        "Check the weather conditions in Chicago",
        "What temperature is it in Los Angeles?",
        "Will it be sunny in Barcelona tomorrow?",
        "How cold is it in Moscow today?",
        "What's the wind speed in San Francisco?",
        "Is there a storm coming to Houston?",
        "Check if it's cloudy in Berlin",
        "What's the UV index in Phoenix?",
        "How hot will it get in Dubai today?",
        "Is it raining in Mumbai right now?",
        "What's the air quality in Beijing?",
        "Will there be fog in London tonight?",
        "Check the weather for my trip to Rome",
        "What should I wear in Toronto today based on weather?",
        "Is it beach weather in Hawaii?",
        "How's the weather in Singapore this week?",
        "What's the precipitation chance in Portland?",
        "Is it freezing in Minneapolis?",
        "Check the forecast for Amsterdam",
        "What's the weather like in Cape Town?",
        "Will it be warm enough for a picnic in Austin?",
        "Is there a heat wave in Phoenix?",
        "What's the pollen count in Atlanta?",
        "How windy is it in Wellington?",
        "Check the visibility in San Diego",
        "Is it muggy in New Orleans?",
        "What's the dew point in Tampa?",
        "Will it snow in Stockholm this weekend?",
        "How's the weather at the Grand Canyon?",
        "Check conditions at Mount Fuji",
        "What's the temperature at Machu Picchu?",
        "Is it monsoon season in Bangkok?",
        "Will I need sunscreen in Cancun?",
        "How's the weather for skiing in Aspen?",
        "Check the surf conditions in Malibu",
        "What's the weather forecast for Edinburgh?",
        "Is it typhoon season in Tokyo?",
        "How's the weather in the Swiss Alps?",
        "What's the temperature in Death Valley?",
        "Check the weather in Reykjavik",
        "Is it dry season in Bali?",
        "What's the forecast for the Sahara Desert?",
    ],
    "search_web": [
        "Find recent news about SpaceX launches",
        "Search for the best restaurants in San Francisco",
        "Look up information about machine learning",
        "Find articles about climate change",
        "Search for reviews of the latest MacBook",
        "Look up the history of the Roman Empire",
        "Find tutorials on React.js",
        "Search for upcoming concerts in my area",
        "Look up the symptoms of the flu",
        "Find recipes for chocolate cake",
        "Search for flights to Japan",
        "Look up the stock price of Apple",
        "Find information about visa requirements for France",
        "Search for the best hiking trails near me",
        "Look up reviews of the new Tesla Model Y",
        "Find news about the latest tech acquisitions",
        "Search for Python programming courses",
        "Look up the population of India",
        "Find information about solar panel installation",
        "Search for job openings in data science",
        "Look up the rules of chess",
        "Find the best credit cards for travel",
        "Search for documentary recommendations",
        "Look up how to start a small business",
        "Find information about the Mars rover",
        "Search for healthy meal prep ideas",
        "Look up the latest COVID-19 guidelines",
        "Find reviews of noise-canceling headphones",
        "Search for yoga classes online",
        "Look up the biography of Elon Musk",
        "Find information about renewable energy",
        "Search for wedding venue ideas",
        "Look up the best books of 2024",
        "Find tutorials on video editing",
        "Search for apartment listings in Boston",
        "Look up how to train for a marathon",
        "Find information about cryptocurrency regulations",
        "Search for the best coffee shops in Portland",
        "Look up symptoms of vitamin D deficiency",
        "Find reviews of electric vehicles",
        "Search for online degree programs",
        "Look up the history of jazz music",
        "Find information about starting a podcast",
        "Search for camping gear recommendations",
        "Look up how to improve credit score",
        "Find the best smartphones under $500",
        "Search for virtual museum tours",
        "Look up tips for learning Spanish",
        "Find information about pet adoption",
        "Search for home workout routines",
    ],
    "read_file": [
        "Show me the contents of config.yaml",
        "Read the README.md file",
        "What's in the settings.json file?",
        "Display the contents of main.py",
        "Open and show me data.csv",
        "Read the log file from yesterday",
        "Show me what's in package.json",
        "Display the contents of .env.example",
        "Read the Dockerfile",
        "What does the Makefile contain?",
        "Show me the gitignore file",
        "Read the requirements.txt",
        "Display the contents of tsconfig.json",
        "What's in the nginx.conf file?",
        "Show me the docker-compose.yml",
        "Read the CHANGELOG.md",
        "Display the LICENSE file",
        "What's in the webpack.config.js?",
        "Show me the contents of index.html",
        "Read the styles.css file",
        "Display what's in app.py",
        "What does the setup.py contain?",
        "Show me the pyproject.toml",
        "Read the Cargo.toml file",
        "Display the go.mod contents",
        "What's in the pom.xml?",
        "Show me the build.gradle file",
        "Read the CMakeLists.txt",
        "Display the .bashrc contents",
        "What's in the .zshrc file?",
        "Show me the ssh config",
        "Read the hosts file",
        "Display the crontab",
        "What's in the supervisord.conf?",
        "Show me the prometheus.yml",
        "Read the grafana dashboard json",
        "Display the terraform.tfvars",
        "What's in the ansible playbook?",
        "Show me the kubernetes deployment yaml",
        "Read the helm values file",
        "Display the circleci config",
        "What's in the github actions workflow?",
        "Show me the jest.config.js",
        "Read the eslintrc file",
        "Display the prettierrc",
        "What's in the babel config?",
        "Show me the rollup.config.js",
        "Read the vite.config.ts",
        "Display the next.config.js",
        "What's in the nuxt.config.ts?",
    ],
    "write_file": [
        "Save this text to notes.txt: Hello world",
        "Create a file called output.json with the results",
        "Write the following to config.yaml: debug: true",
        "Save my notes to diary.txt",
        "Create a new file report.md with the summary",
        "Write the error log to errors.txt",
        "Save the API response to response.json",
        "Create a backup file called data_backup.sql",
        "Write my todo list to tasks.txt",
        "Save the configuration to settings.ini",
        "Create a new Python script called helper.py",
        "Write the test results to test_output.txt",
        "Save the meeting notes to meeting_2024.md",
        "Create an HTML file called index.html",
        "Write the CSS styles to styles.css",
        "Save the SQL query to query.sql",
        "Create a shell script called deploy.sh",
        "Write the environment variables to .env",
        "Save the user data to users.csv",
        "Create a YAML config called app.yml",
        "Write the documentation to docs.md",
        "Save the changelog to CHANGELOG.md",
        "Create a JSON schema file",
        "Write the build output to build.log",
        "Save the analysis results to analysis.txt",
        "Create a requirements file",
        "Write the error messages to debug.log",
        "Save the extracted data to export.json",
        "Create a manifest file",
        "Write the version info to VERSION",
        "Save the credentials template to creds.example",
        "Create a gitignore file",
        "Write the license to LICENSE.md",
        "Save the readme content to README.md",
        "Create a contributing guide",
        "Write the API docs to api.md",
        "Save the schema to schema.json",
        "Create a migration file",
        "Write the seed data to seeds.sql",
        "Save the fixtures to fixtures.json",
        "Create a test data file",
        "Write the mock responses to mocks.json",
        "Save the constants to constants.py",
        "Create a types definition file",
        "Write the interfaces to types.ts",
        "Save the enum definitions to enums.py",
        "Create a utility functions file",
        "Write the helper methods to utils.js",
        "Save the class definitions to models.py",
        "Create a configuration template",
    ],
    "run_code": [
        "Execute this Python code: print('Hello')",
        "Run this script: for i in range(10): print(i)",
        "Calculate 2 + 2 using Python",
        "Execute: import math; print(math.pi)",
        "Run this code to check the result",
        "Execute the fibonacci function",
        "Run this sorting algorithm",
        "Calculate the factorial of 10",
        "Execute this data processing script",
        "Run the test suite",
        "Execute the benchmark code",
        "Run this regex pattern matching",
        "Calculate the mean of this list",
        "Execute the matrix multiplication",
        "Run this API request code",
        "Execute the file parsing script",
        "Run this JSON processing code",
        "Calculate the standard deviation",
        "Execute the string manipulation",
        "Run this list comprehension",
        "Execute the dictionary operations",
        "Run this class instantiation",
        "Calculate the sum of squares",
        "Execute the lambda function",
        "Run this generator expression",
        "Execute the async function",
        "Run this decorator example",
        "Calculate using numpy",
        "Execute the pandas operation",
        "Run this matplotlib plot code",
        "Execute the scikit-learn model",
        "Run this tensorflow prediction",
        "Calculate the correlation",
        "Execute the web scraping code",
        "Run this database query code",
        "Execute the encryption function",
        "Run this hashing algorithm",
        "Calculate the checksum",
        "Execute the compression code",
        "Run this serialization example",
        "Execute the threading code",
        "Run this multiprocessing script",
        "Calculate using decimal precision",
        "Execute the datetime manipulation",
        "Run this timezone conversion",
        "Execute the url parsing code",
        "Run this path manipulation",
        "Calculate the hash of this string",
        "Execute the base64 encoding",
        "Run this binary conversion",
    ],
    "send_email": [
        "Send an email to john@example.com about the meeting",
        "Email my boss that I'll be late today",
        "Send a message to support@company.com asking about my order",
        "Email the team about the project update",
        "Send an email to hr@company.com with my vacation request",
        "Email the client the proposal document",
        "Send a follow-up email to the interviewer",
        "Email my professor about the assignment extension",
        "Send an email to the vendor requesting a quote",
        "Email the IT department about the system issue",
        "Send a thank you email to the recruiter",
        "Email the accountant the expense report",
        "Send an email to schedule a dentist appointment",
        "Email the landlord about the maintenance request",
        "Send a complaint email to customer service",
        "Email the contractor the project specifications",
        "Send an email to RSVP for the event",
        "Email the insurance company about the claim",
        "Send a reminder email about the deadline",
        "Email the travel agent about the booking",
        "Send an email to cancel the subscription",
        "Email the supplier about the delivery status",
        "Send a welcome email to the new team member",
        "Email the sponsor with the event details",
        "Send an email requesting a reference letter",
        "Email the bank about the account issue",
        "Send a birthday greeting email to my colleague",
        "Email the editor about the article submission",
        "Send an email to dispute the charge",
        "Email the realtor about the property listing",
        "Send a meeting invitation email",
        "Email the caterer about the menu options",
        "Send an email to request a refund",
        "Email the coach about practice schedule",
        "Send a newsletter to the mailing list",
        "Email the photographer about the photoshoot",
        "Send an email to introduce myself",
        "Email the organizer about the conference",
        "Send a status update email to stakeholders",
        "Email the lawyer about the contract review",
        "Send an email to negotiate the salary",
        "Email the designer about the logo revisions",
        "Send a farewell email to coworkers",
        "Email the hospital about the appointment",
        "Send an email to request time off",
        "Email the school about the enrollment",
        "Send an apology email to the customer",
        "Email the partner about the collaboration",
        "Send an email to confirm the reservation",
        "Email the manager about the promotion",
    ],
    "get_calendar": [
        "What meetings do I have today?",
        "Show my calendar for next Monday",
        "What's on my schedule for December 15th?",
        "List my appointments for this week",
        "Do I have any events on Friday?",
        "Check my calendar for tomorrow",
        "What's scheduled for next week?",
        "Show me my afternoon appointments",
        "Do I have anything at 3pm?",
        "List all meetings for this month",
        "What's on my schedule for the holidays?",
        "Check if I'm free on Saturday",
        "Show my morning calendar",
        "What events do I have next Wednesday?",
        "List my recurring meetings",
        "Check my availability for Tuesday",
        "What's on my calendar for Q4?",
        "Show me the team meeting schedule",
        "Do I have any conflicts this week?",
        "List all-day events for this month",
        "What's scheduled after lunch today?",
        "Check my calendar for the project deadline",
        "Show my schedule for the conference week",
        "Do I have any evening appointments?",
        "List my video calls for today",
        "What's on my personal calendar?",
        "Check my work schedule for Monday",
        "Show upcoming birthdays on my calendar",
        "Do I have any dentist appointments?",
        "List my calendar invites pending response",
        "What's scheduled for the product launch?",
        "Check my calendar for client meetings",
        "Show my schedule during vacation",
        "Do I have any overlapping meetings?",
        "List my calendar for the sprint review",
        "What's on my schedule for the interview?",
        "Check my availability next Thursday",
        "Show my calendar for performance reviews",
        "Do I have time for a coffee chat?",
        "List my schedule for onboarding week",
        "What's on my calendar for training?",
        "Check my schedule for the workshop",
        "Show my calendar for the hackathon",
        "Do I have any webinars scheduled?",
        "List my calendar for board meetings",
        "What's scheduled for the annual review?",
        "Check my calendar for 1-on-1s",
        "Show my schedule for the offsite",
        "Do I have any deadlines this week?",
        "List my calendar events with reminders",
    ],
    "create_reminder": [
        "Remind me to call mom at 5pm",
        "Set a reminder for the meeting tomorrow at 10am",
        "Create a reminder to buy groceries this evening",
        "Remind me to submit the report by Friday",
        "Set a reminder for my doctor's appointment next week",
        "Remind me to take my medication at 8am",
        "Create a reminder to pay the rent on the 1st",
        "Remind me about the dentist on Thursday",
        "Set a reminder to call the bank tomorrow",
        "Remind me to pick up dry cleaning",
        "Create a reminder for the project deadline",
        "Remind me to water the plants every Sunday",
        "Set a reminder to renew my passport",
        "Remind me about the team lunch on Friday",
        "Create a reminder to backup my files",
        "Remind me to exercise at 6am daily",
        "Set a reminder for the flight check-in",
        "Remind me to cancel the subscription next month",
        "Create a reminder for the warranty expiration",
        "Remind me to review the contract tomorrow",
        "Set a reminder to call the plumber",
        "Remind me about the birthday party Saturday",
        "Create a reminder to update my resume",
        "Remind me to take a break every 2 hours",
        "Set a reminder for the car service appointment",
        "Remind me to send the invoice Monday",
        "Create a reminder to check in on the project",
        "Remind me about the library book due date",
        "Set a reminder to schedule the interview",
        "Remind me to prepare the presentation",
        "Create a reminder for the vaccination appointment",
        "Remind me to order office supplies",
        "Set a reminder to follow up with the client",
        "Remind me about the concert tickets sale",
        "Create a reminder to review the pull request",
        "Remind me to attend the webinar at 2pm",
        "Set a reminder for the lease renewal",
        "Remind me to book the restaurant for dinner",
        "Create a reminder to study for the exam",
        "Remind me about the parent-teacher meeting",
        "Set a reminder to check the test results",
        "Remind me to submit the expense report",
        "Create a reminder for the software license renewal",
        "Remind me to call grandma on her birthday",
        "Set a reminder to change the air filter",
        "Remind me about the charity run registration",
        "Create a reminder to declutter the garage",
        "Remind me to review my investment portfolio",
        "Set a reminder for the eye exam next month",
        "Remind me to send thank you cards",
    ],
    "search_database": [
        "Query the database for all users created last month",
        "Find all orders with status 'pending'",
        "Search the database for products under $50",
        "Get all customers from California",
        "Query for employees in the engineering department",
        "Find all transactions from yesterday",
        "Search for users who haven't logged in recently",
        "Get all products with low inventory",
        "Query for orders placed in December",
        "Find all active subscriptions",
        "Search the database for duplicate entries",
        "Get all invoices that are overdue",
        "Query for customers with premium accounts",
        "Find all shipments with delivery issues",
        "Search for products in the electronics category",
        "Get all users with admin privileges",
        "Query for failed payment attempts",
        "Find all reviews with ratings below 3 stars",
        "Search the database for inactive accounts",
        "Get all orders above $1000",
        "Query for employees hired this year",
        "Find all products that are out of stock",
        "Search for customers by email domain",
        "Get all support tickets marked urgent",
        "Query for cancelled subscriptions",
        "Find all items in the wishlist table",
        "Search the database for seasonal products",
        "Get all users who referred others",
        "Query for orders with discount codes",
        "Find all vendors with overdue payments",
        "Search for products with price changes",
        "Get all log entries with errors",
        "Query for users in the beta program",
        "Find all campaigns that are active",
        "Search the database for archived records",
        "Get all appointments for next week",
        "Query for products with missing images",
        "Find all customers with birthday this month",
        "Search for transactions flagged for review",
        "Get all comments pending moderation",
        "Query for inventory adjustments",
        "Find all returns processed today",
        "Search the database for unverified emails",
        "Get all coupons expiring soon",
        "Query for users by registration date",
        "Find all products with bulk pricing",
        "Search for orders with shipping delays",
        "Get all gift cards with balance",
        "Query for revenue by region",
        "Find all abandoned shopping carts",
    ],
    "translate_text": [
        "Translate 'Hello, how are you?' to Spanish",
        "Convert this text to French: Good morning",
        "Translate 'Thank you very much' to Japanese",
        "How do you say 'Where is the train station?' in German",
        "Translate this paragraph to Chinese",
        "Convert 'I love you' to Italian",
        "Translate the menu to English",
        "How do you say 'Goodbye' in Portuguese?",
        "Translate 'Help!' to Korean",
        "Convert this email to Spanish",
        "Translate the instructions to French",
        "How do you say 'How much does this cost?' in Mandarin",
        "Translate 'Nice to meet you' to Arabic",
        "Convert this document to German",
        "Translate the error message to Japanese",
        "How do you say 'I don't understand' in Russian",
        "Translate 'Where is the bathroom?' to Italian",
        "Convert this recipe to Spanish",
        "Translate the warning label to Chinese",
        "How do you say 'Please wait' in Hindi",
        "Translate this contract to Portuguese",
        "Convert 'Happy Birthday' to French",
        "Translate the user manual to Korean",
        "How do you say 'Sorry' in Dutch",
        "Translate 'Welcome' to Greek",
        "Convert this article to Swedish",
        "Translate the lyrics to English",
        "How do you say 'Cheers' in Polish",
        "Translate 'Good night' to Turkish",
        "Convert this review to Spanish",
        "Translate the sign to English",
        "How do you say 'I'm lost' in Thai",
        "Translate 'Have a nice day' to Vietnamese",
        "Convert this message to Hebrew",
        "Translate the subtitle to Indonesian",
        "How do you say 'Water please' in Czech",
        "Translate 'Can you help me?' to Romanian",
        "Convert this poem to Hungarian",
        "Translate the caption to Finnish",
        "How do you say 'Excuse me' in Norwegian",
        "Translate 'Yes' and 'No' to Danish",
        "Convert this story to Ukrainian",
        "Translate the quote to Swahili",
        "How do you say 'Where can I find...?' in Malay",
        "Translate 'It's delicious' to Tagalog",
        "Convert this proverb to Latin",
        "Translate the speech to Persian",
        "How do you say 'My name is...' in Bengali",
        "Translate 'What time is it?' to Tamil",
        "Convert this phrase to Urdu",
    ],
}


def load_curated_dataset() -> list[EvalPrompt]:
    """
    Load the curated high-quality dataset.

    Returns:
        List of EvalPrompt objects (50 per tool, 500 total)
    """
    dataset = []

    for tool_name, prompts in CURATED_PROMPTS.items():
        for prompt in prompts:
            dataset.append(EvalPrompt(
                prompt=prompt,
                expected_tool=tool_name,
            ))

    return dataset
