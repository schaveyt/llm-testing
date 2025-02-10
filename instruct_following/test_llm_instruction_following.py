import json
import asyncio
import os
import time
import signal
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from statistics import mean, stdev, median
from dotenv import load_dotenv
import aiohttp
from concurrent.futures import TimeoutError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

# Initialize rich console for colored output
console = Console()

# Load environment variables from .env file
load_dotenv()
console.print("[green]Environment loaded[/green]")

# Constants
DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 3

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
        console.print(f"[blue]OpenRouterClient initialized with API key: {api_key[:4]}...[/blue]")
        
    async def generate(self, model: str, messages: list, timeout: int = DEFAULT_TIMEOUT) -> str:
        console.print(f"[cyan]Generating response for model: {model}[/cyan]")
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
        
        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    console.print("[cyan]Making API request...[/cyan]")
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=timeout
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        console.print("[green]API response received[/green]")
                        return data['choices'][0]['message']['content']
            except TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    raise
                console.print(f"[yellow]Timeout occurred. Retrying ({attempt + 2}/{MAX_RETRIES})[/yellow]")
                await asyncio.sleep(1)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    console.print(f"[red]Error in generate: {str(e)}[/red]")
                    raise
                console.print(f"[yellow]Error occurred. Retrying ({attempt + 2}/{MAX_RETRIES})[/yellow]")
                await asyncio.sleep(1)

class TestStatistics:
    """Handles statistical calculations for test results"""
    @staticmethod
    def calculate_success_rate(results: List[dict]) -> float:
        successes = sum(1 for r in results if r["validation"]["valid_xml"] and not r["error"])
        return successes / len(results) if results else 0.0

    @staticmethod
    def calculate_response_times(results: List[dict]) -> Dict[str, float]:
        times = [r.get("response_time", 0) for r in results]
        return {
            "mean": mean(times) if times else 0.0,
            "median": median(times) if times else 0.0,
            "stdev": stdev(times) if len(times) > 1 else 0.0,
            "min": min(times) if times else 0.0,
            "max": max(times) if times else 0.0
        }

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

    async def run_test_case(self, model: str, test_case: dict, pass_number: int, 
                           progress: Progress) -> dict:
        task_id = progress.add_task(
            f"[cyan]Pass {pass_number}/{self.passes}: {model}",
            total=None,
            spinner="dots"
        )
        
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
            progress.update(task_id, completed=True, description=f"[green]✓ {model} ({elapsed_time:.2f}s)")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            progress.update(task_id, completed=True, description=f"[red]✗ {model} ({elapsed_time:.2f}s)")
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

    async def run_all_tests(self):
        console.print("\n[bold green]=== Starting Test Run ===[/bold green]")
        
        total_tests = len(self.test_matrix) * self.passes * len(self.models)
        completed_tests = 0
        
        for i, test_case in enumerate(self.test_matrix):
            console.print(f"\n[bold cyan]Test Case {i+1}/{len(self.test_matrix)}: {test_case['description']}[/bold cyan]")
            
            for pass_num in range(1, self.passes + 1):
                with Progress(
                    SpinnerColumn(),
                    *Progress.get_default_columns(),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True
                ) as progress:
                    results = await asyncio.gather(
                        *[self.run_test_case(model, test_case, pass_num, progress) 
                          for model in self.models]
                    )
                completed_tests += len(self.models)
                console.print(f"Progress: {completed_tests}/{total_tests} tests completed")
                
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

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Instruction Following Test Suite")
    parser.add_argument("--passes", "-p", type=int, default=3,
                      help="Number of test passes to run (default: 3)")
    parser.add_argument("--matrix", "-m", type=str, default="test_matrix.json",
                      help="Path to test matrix JSON file (default: test_matrix.json)")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT,
                      help=f"API request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--output-dir", "-o", type=str, default=".",
                      help="Directory to save result files (default: current directory)")
    return parser.parse_args()

def setup_signal_handlers():
    def handle_interrupt(signum, frame):
        console.print("\n[red]Interrupted by user. Cleaning up...[/red]")
        raise KeyboardInterrupt()
    
    signal.signal(signal.SIGINT, handle_interrupt)

if __name__ == "__main__":
    try:
        setup_signal_handlers()
        args = parse_args()
        
        # Validate and create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\n[bold blue]LLM Instruction Following Test Suite[/bold blue]")
        console.print(f"[blue]Running {args.passes} passes using test matrix: {args.matrix}[/blue]")
        console.print(f"[blue]Results will be saved to: {output_dir}[/blue]")
        
        tester = ToolCallTester(
            test_matrix_path=args.matrix,
            passes=args.passes,
            timeout=args.timeout,
            output_dir=output_dir
        )
        asyncio.run(tester.run_all_tests())
    except KeyboardInterrupt:
        console.print("\n[red]Test run cancelled by user[/red]")
        exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        exit(1)
