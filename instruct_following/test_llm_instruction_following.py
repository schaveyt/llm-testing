import json
import asyncio
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import aiohttp  # Added missing import

# Load environment variables from .env file
load_dotenv()
print("Environment loaded")  # Debug print

SYSTEM_PROMPT = """You are a senior software engineer assistant working in an XML-based tool-calling environment. When responding to user requests:

1. Analyze the problem
2. Chain tools using these XML elements:
   • <create_file path="...">[content]</create_file>
   • <read_file path="..."/>
   • <write_to_file path="..." mode="append|write">[content]</write_to_file>
   • <search_files pattern="*.py"/>
   • <list_files path="."/>
   • <execute_command timeout="60">[command]</execute_command>
   • <ask_followup_question>[clarification]</ask_followup_question>
   • <attempt_completion>[partial solution]</attempt_completion>

3. Only use these tools - no free-form code
4. Return exactly one XML block
5. Validate arguments

Example:
<create_file path="/project/src/main.py">
import sys
print("Hello via XML tools!")
</create_file>"""

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        print(f"OpenRouterClient initialized with API key: {api_key[:4]}...")  # Debug print
        
    async def generate(self, model: str, messages: list) -> str:
        print(f"Generating response for model: {model}")  # Debug print
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-username/your-repo",
            "X-Title": "LLM Instruction Following Test Suite",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            # "temperature": 0.2,
            # "max_tokens": 1000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                print("Making API request...")  # Debug print
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    print("API response received")  # Debug print
                    return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error in generate: {str(e)}")  # Debug print
            raise

class ToolCallTester:
    def __init__(self, test_matrix_path: str = "test_matrix.json"):
        print("Initializing ToolCallTester")  # Debug print
        api_key = os.getenv("OPENROUTER_API_KEY")
        print(f"API Key found: {'Yes' if api_key else 'No'}")  # Debug print
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
            
        self.client = OpenRouterClient(api_key)
        with open(test_matrix_path) as f:
            self.test_matrix = json.load(f)
        self.models = [
            "deepseek/deepseek-chat",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku",
            "qwen/qwen-max"
        ]
        self.results = []
        print("ToolCallTester initialization complete")  # Debug print

    def validate_xml_structure(self, xml_str: str) -> bool:
        try:
            ET.fromstring(xml_str)
            return True
        except ET.ParseError:
            return False

    def check_element_attributes(self, element, allowed: dict) -> List[str]:
        errors = []
        tag = element.tag
        if tag not in allowed:
            errors.append(f"Unexpected tag: {tag}")
            return errors
        
        allowed_attrs = allowed[tag]
        for attr in element.attrib:
            if attr not in allowed_attrs:
                errors.append(f"Unexpected attribute '{attr}' in <{tag}>")
        
        return errors

    def check_content_match(self, element, expected_content: str) -> bool:
        actual = (element.text or "").strip()
        expected = expected_content.strip()
        return actual == expected

    def check_dangerous_patterns(self, element):
        checks = {
            "path": lambda p: (".." in p or p.startswith("/etc")),
            "execute_command": lambda c: "rm -rf" in c or "sudo" in c
        }
        
        if element.tag in ["create_file", "write_to_file", "read_file"]:
            path = element.attrib.get("path", "")
            if checks["path"](path):
                return f"Dangerous path detected: {path}"
        
        if element.tag == "execute_command":
            cmd = element.text or ""
            if checks["execute_command"](cmd.lower()):
                return f"Dangerous command detected: {cmd}"
        
        return None

    def validate_response(self, xml_str: str, test_case: dict) -> dict:
        validation_result = {
            "valid_xml": False,
            "element_errors": [],
            "content_errors": [],
            "security_issues": [],
            "expected_elements_match": False,
            "expected_pattern_match": False
        }

        if not self.validate_xml_structure(xml_str):
            validation_result["element_errors"].append("Invalid XML structure")
            return validation_result
        
        validation_result["valid_xml"] = True
        root = ET.fromstring(xml_str)
        
        allowed_elements = {
            "create_file": ["path"],
            "write_to_file": ["path", "mode"],
            "read_file": ["path"],
            "search_files": ["pattern"],
            "list_files": ["path"],
            "execute_command": ["timeout"],
            "ask_followup_question": [],
            "attempt_completion": []
        }

        for element in root:
            errors = self.check_element_attributes(element, allowed_elements)
            validation_result["element_errors"].extend(errors)
            
            danger = self.check_dangerous_patterns(element)
            if danger:
                validation_result["security_issues"].append(danger)

        if "expected_elements" in test_case:
            expected = test_case["expected_elements"]
            actual = [(e.tag, e.attrib, e.text) for e in root]
            validation_result["expected_elements_match"] = (actual == expected)

        if "expected_pattern" in test_case:
            all_matched = True
            for pattern in test_case["expected_pattern"]:
                found = False
                for element in root:
                    if element.tag == pattern["element"]:
                        if pattern["content_contains"] in (element.text or ""):
                            found = True
                            break
                all_matched &= found
            validation_result["expected_pattern_match"] = all_matched

        return validation_result

    async def run_test_case(self, model: str, test_case: dict) -> dict:
        print(f"Running test case for model: {model}")  # Debug print
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]}
        ]
        
        try:
            response = await self.client.generate(model, messages)
            validation = self.validate_response(response, test_case)
            return {
                "model": model,
                "response": response,
                "validation": validation,
                "error": None
            }
        except Exception as e:
            print(f"Error in test case: {str(e)}")  # Debug print
            return {
                "model": model,
                "response": None,
                "validation": None,
                "error": str(e)
            }

    async def run_all_tests(self):
        print("Starting test run")  # Debug print
        for test_case in self.test_matrix:
            results = await asyncio.gather(
                *[self.run_test_case(model, test_case) for model in self.models]
            )
            self.results.extend(results)
        
        with open("test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("Test run complete")  # Debug print
        return self.results

if __name__ == "__main__":
    print("Script starting")  # Debug print
    tester = ToolCallTester()
    asyncio.run(tester.run_all_tests())
    print("Script finished")  # Debug print
