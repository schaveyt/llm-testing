import json
import time
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import xml.etree.ElementTree as ET
from rich.console import Console

from src.api_client import OpenRouterClient, DEFAULT_TIMEOUT
from src.test_statistics import TestStatistics

# Initialize rich console for colored output
console = Console()

def load_system_prompt():
    """Load the system prompt from system_prompt.md file."""
    with open("system_prompt.md", "r") as f:
        return f.read()

SYSTEM_PROMPT = load_system_prompt()

class ToolCallTester:
    def __init__(self, test_matrix_path: str = "test_matrix.json", passes: int = 3, 
                 timeout: int = DEFAULT_TIMEOUT, output_dir: Path = Path(".")):
        console.print("\n[bold green]=== Initializing ToolCallTester ===[/bold green]")
        self.passes = passes
        self.timeout = timeout
        self.output_dir = output_dir
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
            
        self.client = OpenRouterClient(api_key)
        
        # Validate test matrix path
        if not Path(test_matrix_path).exists():
            raise FileNotFoundError(f"Test matrix file not found: {test_matrix_path}")
        with open(test_matrix_path) as f:
            self.test_matrix = json.load(f)
        self.models = [
            "deepseek/deepseek-chat",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku",
            "qwen/qwen-max"
        ]
        self.results = {
            "test_configuration": {
                "total_passes": passes,
                "timestamp": datetime.now().isoformat(),
                "models": self.models
            },
            "results": {},
            "summary": {}
        }
        print(f"Configuration loaded: {passes} passes for {len(self.models)} models")

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

    async def run_test_case(self, model: str, test_case: dict, pass_number: int) -> dict:
        start_time = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]}
        ]
        
        try:
            response = await self.client.generate(model, messages, timeout=self.timeout)
            validation = self.validate_response(response, test_case)
            elapsed_time = time.time() - start_time
            result = {
                "pass_number": pass_number,
                "model": model,
                "response": response,
                "validation": validation,
                "error": None,
                "response_time": elapsed_time
            }
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "pass_number": pass_number,
                "model": model,
                "response": None,
                "validation": None,
                "error": str(e),
                "response_time": elapsed_time
            }

    def calculate_test_statistics(self, model_results: List[dict]) -> dict:
        success_rate = TestStatistics.calculate_success_rate(model_results)
        time_stats = TestStatistics.calculate_response_times(model_results)
        return {
            "success_rate": success_rate,
            "avg_response_time": time_stats["mean"],
            "response_time_median": time_stats["median"],
            "response_time_stdev": time_stats["stdev"],
            "min_response_time": time_stats["min"],
            "max_response_time": time_stats["max"],
            "total_passes": len(model_results)
        }

    async def run_all_tests(self, progress_callback: Optional[Callable[[str], None]] = None):
        console.print("\n[bold green]=== Starting Test Run ===[/bold green]")
        
        for i, test_case in enumerate(self.test_matrix):
            console.print(f"\n[bold cyan]Test Case {i+1}/{len(self.test_matrix)}: {test_case['description']}[/bold cyan]")
            
            for pass_num in range(1, self.passes + 1):
                results = await asyncio.gather(
                    *[self.run_test_case(model, test_case, pass_num) 
                      for model in self.models]
                )
                
                if progress_callback:
                    desc = f"Test {i+1}/{len(self.test_matrix)}, Pass {pass_num}/{self.passes}"
                    progress_callback(desc)
                
                # Organize results by model
                for result in results:
                    model = result["model"]
                    if model not in self.results["results"]:
                        self.results["results"][model] = {}
                    
                    test_idx = str(i)
                    if test_idx not in self.results["results"][model]:
                        self.results["results"][model][test_idx] = {
                            "description": test_case["description"],
                            "passes": []
                        }
                    
                    self.results["results"][model][test_idx]["passes"].append(result)

        # Calculate statistics
        for model in self.models:
            model_results = []
            for test_idx in self.results["results"][model]:
                passes = self.results["results"][model][test_idx]["passes"]
                model_results.extend(passes)
                self.results["results"][model][test_idx]["statistics"] = \
                    self.calculate_test_statistics(passes)
            
            self.results["summary"][model] = self.calculate_test_statistics(model_results)

        # Save detailed JSON results
        json_path = self.output_dir / "test_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Create detailed CSV file
        detailed_csv_path = self.output_dir / "test_results_detailed.csv"
        with open(detailed_csv_path, "w") as f:
            f.write("model,test_case,pass_number,valid_xml,element_errors,security_issues," +
                   "expected_elements_match,expected_pattern_match,has_error,response_time\n")
            
            for model in self.results["results"]:
                for test_idx, test_data in self.results["results"][model].items():
                    for pass_result in test_data["passes"]:
                        validation = pass_result["validation"] if pass_result["validation"] else {}
                        row = [
                            model,
                            test_data["description"].replace(",", ";"),
                            str(pass_result["pass_number"]),
                            str(validation.get("valid_xml", False)),
                            str(len(validation.get("element_errors", []))),
                            str(len(validation.get("security_issues", []))),
                            str(validation.get("expected_elements_match", False)),
                            str(validation.get("expected_pattern_match", False)),
                            str(bool(pass_result["error"])),
                            f"{pass_result.get('response_time', 0):.2f}"
                        ]
                        f.write(",".join(row) + "\n")
        
        # Create summary CSV file
        summary_csv_path = self.output_dir / "test_results_summary.csv"
        with open(summary_csv_path, "w") as f:
            f.write("model,success_rate,avg_response_time,response_time_stdev,total_passes\n")
            for model, stats in self.results["summary"].items():
                row = [
                    model,
                    f"{stats['success_rate']:.3f}",
                    f"{stats['avg_response_time']:.2f}",
                    f"{stats['response_time_stdev']:.2f}",
                    str(stats['total_passes'])
                ]
                f.write(",".join(row) + "\n")
        
        console.print("\n[bold green]=== Test Run Complete ===[/bold green]")
        console.print(f"[blue]Results saved to:[/blue]")
        console.print(f"  • {json_path}")
        console.print(f"  • {detailed_csv_path}")
        console.print(f"  • {summary_csv_path}")
        return self.results
