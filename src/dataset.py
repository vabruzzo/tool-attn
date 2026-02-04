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
        "Write to output.json: {\"status\": \"success\"}",
        "Write the following to config.yaml: debug: true",
        "Save to diary.txt: Today was productive",
        "Write to report.md: # Summary",
    ],
    "run_code": [
        "Execute this Python code: print('Hello')",
        "Run this script: for i in range(10): print(i)",
        "Execute this code: print(2 + 2)",
        "Run: import math; print(math.pi)",
        "Execute: print(sum([1, 2, 3, 4, 5]))",
    ],
    "send_email": [
        "Send an email to john@example.com with subject 'Meeting' and body 'See you at 2pm'",
        "Email boss@company.com with subject 'Running Late' saying 'Be there in 30 min'",
        "Send to support@company.com with subject 'Order #123' and body 'What is the status?'",
        "Email team@company.com with subject 'Update' and body 'Sprint completed'",
        "Send an email to hr@company.com with subject 'Vacation' and body 'Requesting Dec 20-27 off'",
    ],
    "get_calendar": [
        "What meetings do I have today?",
        "Show my calendar for next Monday",
        "What's on my schedule for December 15th?",
        "List my appointments for this week",
        "Do I have any events on Friday?",
    ],
    "create_reminder": [
        "Remind me to call mom at 5:00 PM today",
        "Set a reminder for tomorrow at 10:00 AM: team meeting",
        "Create a reminder for 6:00 PM today: buy groceries",
        "Remind me at 9:00 AM Friday to submit the report",
        "Set a reminder for next Tuesday at 2:30 PM: doctor appointment",
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
        "Translate 'I need help' to Chinese",
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
        "Save this text to notes.txt: Hello world, this is my first note",
        "Write to output.json: {\"status\": \"success\", \"count\": 42}",
        "Write the following to config.yaml: debug: true",
        "Save to diary.txt: Today was a productive day at work",
        "Write to report.md: # Summary\\n\\nProject completed on time",
        "Write to errors.txt: ERROR: Connection timeout at 10:30 AM",
        "Save to response.json: {\"data\": [{\"id\": 1, \"name\": \"test\"}]}",
        "Write to data_backup.sql: INSERT INTO users VALUES (1, 'admin');",
        "Save to tasks.txt: 1. Review code\\n2. Write tests\\n3. Deploy",
        "Write to settings.ini: [app]\\nmode=production\\nport=8080",
        "Write to helper.py: def greet(name):\\n    return f'Hello {name}'",
        "Save to test_output.txt: All 15 tests passed in 2.3 seconds",
        "Write to meeting_2024.md: # Meeting Notes\\n\\n- Discussed roadmap",
        "Write to index.html: <!DOCTYPE html>\\n<html><body>Hello</body></html>",
        "Save to styles.css: body { margin: 0; font-family: sans-serif; }",
        "Write to query.sql: SELECT * FROM users WHERE active = true;",
        "Write to deploy.sh: #!/bin/bash\\necho 'Deploying...'\\nnpm run build",
        "Save to .env: DATABASE_URL=postgres://localhost:5432/mydb",
        "Write to users.csv: id,name,email\\n1,John,john@example.com",
        "Write to app.yml: name: myapp\\nversion: 1.0.0\\nport: 3000",
        "Save to docs.md: # Documentation\\n\\n## Getting Started",
        "Write to CHANGELOG.md: ## v1.0.0\\n\\n- Initial release",
        "Write to schema.json: {\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"integer\"}}}",
        "Save to build.log: Build completed successfully at 14:30:00",
        "Write to analysis.txt: Results: mean=45.2, std=12.3, n=100",
        "Write to requirements.txt: flask==2.0.0\\nrequests==2.28.0",
        "Save to debug.log: DEBUG: Processing started at 09:00:00",
        "Write to export.json: [{\"id\": 1}, {\"id\": 2}, {\"id\": 3}]",
        "Write to manifest.json: {\"name\": \"MyApp\", \"version\": \"1.0\"}",
        "Save to VERSION: 2.1.0",
        "Write to creds.example: API_KEY=your_key_here\\nSECRET=your_secret",
        "Write to .gitignore: node_modules/\\n.env\\n*.log",
        "Save to LICENSE.md: MIT License\\n\\nCopyright 2024",
        "Write to README.md: # My Project\\n\\nA simple application",
        "Write to CONTRIBUTING.md: # How to Contribute\\n\\n1. Fork the repo",
        "Save to api.md: # API Reference\\n\\n## GET /users",
        "Write to schema.sql: CREATE TABLE users (id INT PRIMARY KEY);",
        "Write to migration.sql: ALTER TABLE users ADD COLUMN email VARCHAR;",
        "Save to seeds.sql: INSERT INTO products VALUES (1, 'Widget', 9.99);",
        "Write to fixtures.json: {\"users\": [{\"id\": 1, \"name\": \"Test User\"}]}",
        "Write to testdata.json: {\"input\": \"hello\", \"expected\": \"HELLO\"}",
        "Save to mocks.json: {\"GET /api\": {\"status\": 200, \"body\": {}}}",
        "Write to constants.py: MAX_RETRIES = 3\\nTIMEOUT = 30",
        "Write to types.d.ts: interface User { id: number; name: string; }",
        "Save to types.ts: export type Status = 'active' | 'inactive';",
        "Write to enums.py: from enum import Enum\\nclass Status(Enum):\\n    ACTIVE = 1",
        "Write to utils.py: def slugify(s): return s.lower().replace(' ', '-')",
        "Save to utils.js: export const capitalize = s => s[0].toUpperCase() + s.slice(1);",
        "Write to models.py: class User:\\n    def __init__(self, name):\\n        self.name = name",
        "Write to config.template.yaml: database:\\n  host: localhost\\n  port: 5432",
    ],
    "run_code": [
        "Execute this Python code: print('Hello, World!')",
        "Run this script: for i in range(10): print(i)",
        "Execute this code: print(2 + 2)",
        "Run: import math; print(math.pi)",
        "Execute this Python: print(sum([1, 2, 3, 4, 5]))",
        "Run this code: def fib(n): return n if n < 2 else fib(n-1) + fib(n-2); print(fib(10))",
        "Execute: sorted([3, 1, 4, 1, 5, 9, 2, 6])",
        "Run this: import math; print(math.factorial(10))",
        "Execute this code: data = [1,2,3]; print([x*2 for x in data])",
        "Run: import subprocess; subprocess.run(['pytest', 'tests/'])",
        "Execute this benchmark: import time; start=time.time(); sum(range(1000000)); print(time.time()-start)",
        "Run this code: import re; print(re.findall(r'\\d+', 'abc123def456'))",
        "Execute: import statistics; print(statistics.mean([1, 2, 3, 4, 5]))",
        "Run this: import numpy as np; a=np.array([[1,2],[3,4]]); print(np.dot(a,a))",
        "Execute this code: import requests; r=requests.get('https://api.github.com'); print(r.status_code)",
        "Run: with open('data.txt') as f: print(f.read())",
        "Execute this: import json; print(json.dumps({'key': 'value'}))",
        "Run this code: import statistics; print(statistics.stdev([1, 2, 3, 4, 5]))",
        "Execute: s='hello'; print(s.upper().replace('L', 'X'))",
        "Run this: squares = [x**2 for x in range(10)]; print(squares)",
        "Execute this code: d={'a':1,'b':2}; d['c']=3; print(d)",
        "Run: class Dog: def bark(self): return 'woof'; print(Dog().bark())",
        "Execute this: print(sum(x**2 for x in range(1, 11)))",
        "Run this code: double = lambda x: x*2; print(double(5))",
        "Execute: gen = (x**2 for x in range(5)); print(list(gen))",
        "Run this: import asyncio; async def hello(): return 'hi'; print(asyncio.run(hello()))",
        "Execute this decorator example: @lambda f: lambda x: f(x)*2; def inc(x): return x+1; print(inc(3))",
        "Run: import numpy as np; print(np.mean([1, 2, 3, 4, 5]))",
        "Execute this: import pandas as pd; df=pd.DataFrame({'a':[1,2,3]}); print(df.sum())",
        "Run this code: import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.savefig('plot.png')",
        "Execute: from sklearn.linear_model import LinearRegression; print('sklearn loaded')",
        "Run this: import tensorflow as tf; print(tf.__version__)",
        "Execute this code: import numpy as np; print(np.corrcoef([1,2,3], [4,5,6]))",
        "Run: from bs4 import BeautifulSoup; print(BeautifulSoup('<p>hi</p>', 'html.parser').text)",
        "Execute this: import sqlite3; conn=sqlite3.connect(':memory:'); print('connected')",
        "Run this code: from cryptography.fernet import Fernet; print(Fernet.generate_key()[:20])",
        "Execute: import hashlib; print(hashlib.sha256(b'hello').hexdigest()[:16])",
        "Run this: import zlib; print(len(zlib.compress(b'hello world')))",
        "Execute this code: import gzip; print(gzip.compress(b'test data')[:10])",
        "Run: import pickle; print(pickle.dumps({'key': 'value'})[:20])",
        "Execute this: import threading; t=threading.Thread(target=print, args=('hello',)); t.start()",
        "Run this code: from multiprocessing import Pool; print('pool ready')",
        "Execute: from decimal import Decimal; print(Decimal('0.1') + Decimal('0.2'))",
        "Run this: from datetime import datetime; print(datetime.now().strftime('%Y-%m-%d'))",
        "Execute this code: import pytz; print(pytz.timezone('US/Eastern'))",
        "Run: from urllib.parse import urlparse; print(urlparse('https://example.com/path').netloc)",
        "Execute this: from pathlib import Path; print(Path.home())",
        "Run this code: import hashlib; print(hashlib.md5(b'test').hexdigest())",
        "Execute: import base64; print(base64.b64encode(b'hello world'))",
        "Run this: print(bin(42), hex(42), oct(42))",
    ],
    "send_email": [
        "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Hi John, can we meet at 2pm?'",
        "Email boss@company.com with subject 'Running Late' saying 'I'll be 30 minutes late due to traffic'",
        "Send a message to support@company.com with subject 'Order #12345 Status' asking 'What is my order status?'",
        "Email team@company.com with subject 'Project Update' and body 'Sprint completed successfully'",
        "Send an email to hr@company.com with subject 'Vacation Request' and body 'Requesting Dec 20-27 off'",
        "Email client@business.com with subject 'Proposal' and body 'Please find attached our proposal'",
        "Send an email to recruiter@company.com with subject 'Thank You' and body 'Thanks for the interview'",
        "Email professor@university.edu with subject 'Extension Request' and body 'May I have 2 extra days?'",
        "Send an email to vendor@supplies.com with subject 'Quote Request' and body 'Please quote 100 units'",
        "Email it@company.com with subject 'System Down' and body 'The server is not responding'",
        "Send an email to hiring@company.com with subject 'Application Follow-up' and body 'Checking on my application status'",
        "Email accounting@company.com with subject 'Expense Report' and body 'Attached is my expense report for review'",
        "Send an email to dentist@clinic.com with subject 'Appointment Request' and body 'Need a cleaning next week'",
        "Email landlord@property.com with subject 'Maintenance Request' and body 'The kitchen faucet is leaking'",
        "Send an email to service@company.com with subject 'Complaint' and body 'Product arrived damaged'",
        "Email contractor@build.com with subject 'Project Specs' and body 'Here are the specifications'",
        "Send an email to events@company.com with subject 'RSVP' and body 'I will attend the event'",
        "Email claims@insurance.com with subject 'Claim #98765' and body 'Following up on my claim'",
        "Send an email to team@company.com with subject 'Deadline Reminder' and body 'Report due Friday'",
        "Email travel@agency.com with subject 'Booking Inquiry' and body 'Need flights to Paris'",
        "Send an email to billing@service.com with subject 'Cancel Subscription' and body 'Please cancel my account'",
        "Email supplier@vendor.com with subject 'Delivery Status' and body 'Where is order #5678?'",
        "Send an email to newjoin@company.com with subject 'Welcome!' and body 'Welcome to the team!'",
        "Email sponsor@corp.com with subject 'Event Details' and body 'Event is on March 15th at 6pm'",
        "Send an email to professor@school.edu with subject 'Reference Request' and body 'Would you write a reference?'",
        "Email support@bank.com with subject 'Account Issue' and body 'Cannot access online banking'",
        "Send an email to colleague@work.com with subject 'Happy Birthday!' and body 'Wishing you all the best!'",
        "Email editor@magazine.com with subject 'Article Submission' and body 'Please review my attached article'",
        "Send an email to billing@company.com with subject 'Dispute Charge' and body 'I did not authorize this charge'",
        "Email agent@realty.com with subject 'Property Inquiry' and body 'Is 123 Main St still available?'",
        "Send an email to attendees@company.com with subject 'Meeting Invite' and body 'Please join us Tuesday at 10am'",
        "Email catering@food.com with subject 'Menu Options' and body 'Need menu for 50 guests'",
        "Send an email to refunds@store.com with subject 'Refund Request' and body 'Order #1234 was defective'",
        "Email coach@gym.com with subject 'Practice Schedule' and body 'What time is practice Saturday?'",
        "Send an email to subscribers@list.com with subject 'Newsletter' and body 'Here is this week update'",
        "Email photo@studio.com with subject 'Photoshoot' and body 'Available next Sunday?'",
        "Send an email to contact@company.com with subject 'Introduction' and body 'I am a software engineer interested in opportunities'",
        "Email organizer@conf.com with subject 'Conference Inquiry' and body 'How do I register?'",
        "Send an email to stakeholders@company.com with subject 'Status Update' and body 'Project is on track'",
        "Email lawyer@firm.com with subject 'Contract Review' and body 'Please review the attached contract'",
        "Send an email to hr@company.com with subject 'Salary Discussion' and body 'Can we discuss compensation?'",
        "Email designer@agency.com with subject 'Logo Feedback' and body 'Please make the text larger'",
        "Send an email to team@company.com with subject 'Farewell' and body 'It has been great working with you all'",
        "Email reception@hospital.com with subject 'Appointment' and body 'Need to reschedule my appointment'",
        "Send an email to manager@company.com with subject 'Time Off Request' and body 'Requesting Friday off'",
        "Email admissions@school.edu with subject 'Enrollment Question' and body 'What documents are needed?'",
        "Send an email to customer@client.com with subject 'Apology' and body 'Sorry for the delayed response'",
        "Email partner@business.com with subject 'Collaboration' and body 'Interested in partnering'",
        "Send an email to bookings@hotel.com with subject 'Reservation Confirmation' and body 'Confirming my stay Dec 1-3'",
        "Email supervisor@company.com with subject 'Promotion Discussion' and body 'Can we discuss my career path?'",
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
        "Remind me to call mom at 5:00 PM today",
        "Set a reminder for tomorrow at 10:00 AM: team meeting",
        "Create a reminder for 6:00 PM today: buy groceries",
        "Remind me at 9:00 AM Friday to submit the report",
        "Set a reminder for next Tuesday at 2:30 PM: doctor appointment",
        "Remind me at 8:00 AM every day to take medication",
        "Create a reminder for the 1st of the month at 9:00 AM: pay rent",
        "Remind me Thursday at 3:00 PM: dentist appointment",
        "Set a reminder for tomorrow at 10:00 AM to call the bank",
        "Remind me at 5:30 PM today to pick up dry cleaning",
        "Create a reminder for Friday at 5:00 PM: project deadline",
        "Remind me every Sunday at 10:00 AM to water the plants",
        "Set a reminder for March 1st at 9:00 AM: renew passport",
        "Remind me Friday at 12:00 PM: team lunch",
        "Create a reminder for Sunday at 8:00 PM: backup files",
        "Remind me at 6:00 AM every day to exercise",
        "Set a reminder for 24 hours before flight: check in online",
        "Remind me on the 15th at 9:00 AM to cancel subscription",
        "Create a reminder for December 31st at 10:00 AM: warranty expires",
        "Remind me tomorrow at 2:00 PM to review the contract",
        "Set a reminder for Monday at 9:00 AM to call the plumber",
        "Remind me Saturday at 4:00 PM: birthday party",
        "Create a reminder for this weekend at 10:00 AM: update resume",
        "Remind me every 2 hours starting at 9:00 AM to take a break",
        "Set a reminder for next Monday at 8:00 AM: car service",
        "Remind me Monday at 9:00 AM to send the invoice",
        "Create a reminder for Wednesday at 3:00 PM: check project status",
        "Remind me in 2 weeks at 10:00 AM: library book due",
        "Set a reminder for tomorrow at 11:00 AM to schedule interview",
        "Remind me Thursday at 6:00 PM to prepare presentation",
        "Create a reminder for next Friday at 9:00 AM: vaccination appointment",
        "Remind me Monday at 2:00 PM to order office supplies",
        "Set a reminder for 3 days from now at 10:00 AM: follow up with client",
        "Remind me tomorrow at 10:00 AM: concert tickets go on sale",
        "Create a reminder for today at 4:00 PM: review pull request",
        "Remind me today at 2:00 PM to attend the webinar",
        "Set a reminder for 30 days from now at 9:00 AM: lease renewal",
        "Remind me Friday at 11:00 AM to book restaurant for Saturday dinner",
        "Create a reminder for next week at 7:00 PM every day: study for exam",
        "Remind me Tuesday at 3:30 PM: parent-teacher meeting",
        "Set a reminder for Friday at 10:00 AM to check test results",
        "Remind me end of month at 5:00 PM to submit expense report",
        "Create a reminder for January 15th at 9:00 AM: software license renewal",
        "Remind me on March 5th at 10:00 AM to call grandma for her birthday",
        "Set a reminder for the 1st of next month at 10:00 AM: change air filter",
        "Remind me next Saturday at 8:00 AM: charity run registration deadline",
        "Create a reminder for this Sunday at 9:00 AM: declutter garage",
        "Remind me quarterly at 10:00 AM to review investment portfolio",
        "Set a reminder for the 15th of next month at 9:00 AM: eye exam",
        "Remind me this weekend at 11:00 AM to send thank you cards",
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
