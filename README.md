# ReAct Agent from Scratch

A dependency-free Python implementation of the ReAct (Reasoning + Acting) agent loop. It lets a model choose a tool, records the tool result as an observation, and continues until the model returns a final answer or the step limit is reached. Included tools are a safe arithmetic calculator, keyword search over a bundled knowledge file, and a sandboxed local file reader.

## Features

- Explicit action/observation trace for every tool call.
- Offline demo mode that runs without an API key or network access.
- Optional OpenAI-compatible chat-completions adapter, configured through environment variables.
- Arithmetic evaluation via a restricted Python AST, not `eval()`.
- File reads are restricted to the bundled `data/` directory.
- Standard-library-only tests.

## Requirements

- Python 3.10 or newer.
- No third-party packages are needed for the offline demo or tests.
- Optional API mode needs an API key for the chosen OpenAI-compatible service.

## Setup

```bash
git clone https://github.com/mohan843/react-agent-from-scratch.git
cd react-agent-from-scratch
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Usage

Run the offline demo (default):

```bash
python react_agent.py
```

Ask a different question. The bundled corpus is searched when the demo model selects `search`; simple arithmetic questions use the calculator:

```bash
python react_agent.py "How does a ReAct agent use tools?"
python react_agent.py "What is 12 * (3 + 1)?"
```

The output is JSON containing the final answer and the full action/input/observation trace. To use an OpenAI-compatible model instead, set configuration and enable API mode:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"        # optional
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
python react_agent.py --api "Search the knowledge file for ReAct"
```

The model must follow the prompt contract and return a JSON object per turn, either ` {"action":"tool_name","input":"..."}` or ` {"final":"..."}`. Tool names: `calculator`, `search`, and `read_file`. Set a different maximum tool count with `--max-steps 3`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project layout

```text
react-agent-from-scratch/
├── data/knowledge.txt
├── tests/test_react_agent.py
├── react_agent.py
├── requirements.txt
└── README.md
```

## Safety note

The included file reader only accesses files inside `data/`. The calculator accepts numeric literals and a small set of arithmetic operators. API mode sends the prompt (which includes the question and tool observations) to the provider configured by you; never include secrets in a question.
