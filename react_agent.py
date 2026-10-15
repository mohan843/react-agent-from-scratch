"""Dependency-free ReAct agent loop with safe local tools and an offline demo."""
import argparse
import ast
import json
import re
from pathlib import Path


class Calculator:
    """Evaluate simple arithmetic using an allow-listed AST (never eval)."""
    OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
           ast.Pow: lambda a, b: a ** b}
    UNARY = {ast.UAdd: lambda x: x, ast.USub: lambda x: -x}

    def __call__(self, expression):
        if len(expression) > 120:
            raise ValueError("expression is too long")
        value = self._eval(ast.parse(expression, mode="eval").body)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    def _eval(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self.OPS:
            a, b = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 8:
                raise ValueError("exponent magnitude must be at most 8")
            value = self.OPS[type(node.op)](a, b)
            if abs(value) > 10**12:
                raise ValueError("result is too large")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY:
            return self.UNARY[type(node.op)](self._eval(node.operand))
        raise ValueError("only numeric arithmetic is allowed")


class Search:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, query):
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        lines = [line.strip() for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ranked = sorted(enumerate(lines), key=lambda pair: (
            -len(terms & set(re.findall(r"[a-z0-9]+", pair[1].lower()))), pair[0]))
        hits = [line for _, line in ranked if terms & set(re.findall(r"[a-z0-9]+", line.lower()))][:3]
        return "\n".join(hits) if hits else "No matching information found in the local knowledge file."


class ReadFile:
    def __init__(self, data_dir):
        self.root = Path(data_dir).resolve()

    def __call__(self, name):
        path = (self.root / name).resolve()
        if self.root not in path.parents:
            raise ValueError("file must stay inside the data directory")
        if not path.is_file():
            raise FileNotFoundError(name)
        return path.read_text(encoding="utf-8")[:4000]


class ReActAgent:
    """Ask a model for a tool action, observe the result, repeat, or finish."""
    def __init__(self, model, tools, max_steps=5):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.model, self.tools, self.max_steps = model, tools, max_steps

    def run(self, question):
        if not question.strip():
            raise ValueError("question cannot be empty")
        trace = []
        for _ in range(self.max_steps):
            history = "\n".join(
                f"Action: {x['action']}[{x['input']}]\nObservation: {x['observation']}" for x in trace
            ) or "(no actions yet)"
            prompt = ("You are a ReAct agent. Reply with one JSON object: "
                      '{"action":"tool_name","input":"..."} or {"final":"answer"}. '
                      "Do not invent observations.\nTools: calculator, search, read_file\n"
                      f"Question: {question}\nTrajectory:\n{history}")
            try:
                reply = json.loads(self.model.generate(prompt))
            except (json.JSONDecodeError, TypeError) as exc:
                return {"answer": f"Invalid model response: {exc}", "trace": trace, "ok": False}
            if not isinstance(reply, dict):
                return {"answer": "Model response must be a JSON object.", "trace": trace, "ok": False}
            if "final" in reply:
                return {"answer": str(reply["final"]), "trace": trace, "ok": True}
            name, value = reply.get("action"), reply.get("input", "")
            if not isinstance(name, str) or not isinstance(value, str):
                return {"answer": "Action and input must be strings.", "trace": trace, "ok": False}
            try:
                observation = self.tools[name](value)
            except KeyError:
                observation = f"Unknown tool {name!r}. Available tools: {', '.join(self.tools)}"
            except Exception as exc:
                observation = f"Tool error: {exc}"
            trace.append({"action": name, "input": value, "observation": str(observation)})
        return {"answer": "Stopped at the maximum tool-step limit.", "trace": trace, "ok": False}


class DemoModel:
    """Offline policy for showcasing tool selection and observation grounding."""
    def generate(self, prompt):
        observations = re.findall(r"Observation: (.*)", prompt)
        if observations:
            return json.dumps({"final": f"Based on the tool observation: {observations[-1]}"})
        question = re.search(r"Question: (.*)\nTrajectory:", prompt).group(1)
        expression = re.search(
            r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?[\d\s().*/%+-]*", question
        )
        return json.dumps({"action": "calculator", "input": expression.group(0)} if expression
                          else {"action": "search", "input": question})


def build_agent(data_dir, model, max_steps=5):
    return ReActAgent(model, {
        "calculator": Calculator(),
        "search": Search(Path(data_dir) / "knowledge.txt"),
        "read_file": ReadFile(data_dir),
    }, max_steps)


def main():
    parser = argparse.ArgumentParser(description="Run the offline ReAct agent demo.")
    parser.add_argument("question", nargs="?", default="How does a ReAct agent use tools?")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    result = build_agent(Path(__file__).parent / "data", DemoModel(), args.max_steps).run(args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
