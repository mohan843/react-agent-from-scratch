# ReAct Agent from Scratch

A mini-project implementing the ReAct (Reasoning + Acting) loop in plain Python. The agent picks a tool, records its observation, and then produces a grounded answer. It includes a deterministic offline demo model plus three local tools: a safe calculator, keyword search over sample knowledge, and a data-file reader.

## Requirements

- Python 3.10 or newer
- No third-party packages required

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

Run a question against the bundled knowledge file:

```bash
python react_agent.py
python react_agent.py "How does a ReAct agent use tools?"
```

Try calculator tool selection:

```bash
python react_agent.py "What is 12 * (3 + 1)?"
```

The command prints JSON with an answer and an action/input/observation trace. The offline demo requires no API key or network. `--max-steps 3` changes the maximum number of tool actions.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Files

- `react_agent.py`: agent loop, three tools, and command-line entry point.
- `data/knowledge.txt`: sample source text for local retrieval.
- `tests/test_react_agent.py`: unit tests for tools, safety, and agent behavior.
- `requirements.txt`: documents that there are no external dependencies.

## Safety

Arithmetic is evaluated with a restricted AST, not `eval()`. File reads are confined to the bundled `data/` directory. This project uses a deterministic local demo policy; replace `DemoModel` with your own model adapter to connect an LLM.
