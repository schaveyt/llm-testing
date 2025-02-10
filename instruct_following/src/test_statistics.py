from typing import Dict, List
from statistics import mean, stdev, median

class TestStatistics:
    """Handles statistical calculations for test results"""
    @staticmethod
    def is_test_successful(pass_data: dict) -> bool:
        return (pass_data.get("validation") and 
                pass_data["validation"].get("valid_xml") and 
                not pass_data.get("error"))
    
    @staticmethod
    def calculate_success_rate(results: List[dict]) -> float:
        successes = sum(1 for r in results if TestStatistics.is_test_successful(r))
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
