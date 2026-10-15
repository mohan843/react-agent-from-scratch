import unittest
from pathlib import Path

from react_agent import Calculator, DemoModel, ReadFile, ReActAgent, Search, build_agent

DATA = Path(__file__).parents[1] / "data"


class CalculatorTests(unittest.TestCase):
    def test_precedence_and_parentheses(self):
        self.assertEqual(Calculator()("12 * (3 + 1)"), "48")

    def test_division(self):
        self.assertEqual(Calculator()("7 / 2"), "3.5")

    def test_rejects_code(self):
        with self.assertRaises(ValueError):
            Calculator()("__import__('os').system('echo unsafe')")

    def test_limits_exponent(self):
        with self.assertRaises(ValueError):
            Calculator()("2 ** 9")


class ToolTests(unittest.TestCase):
    def test_search(self):
        self.assertIn("Reasoning and Acting", Search(DATA / "knowledge.txt")("What is ReAct?"))

    def test_reader(self):
        self.assertIn("Reasoning and Acting", ReadFile(DATA)("knowledge.txt"))

    def test_reader_blocks_traversal(self):
        with self.assertRaises(ValueError):
            ReadFile(DATA)("../react_agent.py")


class AgentTests(unittest.TestCase):
    def test_demo_completes_search_loop(self):
        result = build_agent(DATA, DemoModel()).run("How does a ReAct agent use tools?")
        self.assertTrue(result["ok"])
        self.assertEqual(result["trace"][0]["action"], "search")
        self.assertIn("Reasoning and Acting", result["answer"])

    def test_unknown_tool_becomes_observation(self):
        class Model:
            def __init__(self): self.calls = 0
            def generate(self, prompt):
                self.calls += 1
                return '{"action":"missing","input":"x"}' if self.calls == 1 else '{"final":"handled"}'
        result = ReActAgent(Model(), {"calculator": Calculator()}).run("test")
        self.assertIn("Unknown tool", result["trace"][0]["observation"])
        self.assertTrue(result["ok"])

    def test_max_steps(self):
        class Model:
            def generate(self, prompt): return '{"action":"calculator","input":"1+1"}'
        result = ReActAgent(Model(), {"calculator": Calculator()}, max_steps=2).run("test")
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["trace"]), 2)

    def test_empty_question(self):
        with self.assertRaises(ValueError):
            build_agent(DATA, DemoModel()).run(" ")


if __name__ == "__main__":
    unittest.main()
