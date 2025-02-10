# LLM Instruction Following Test Suite

A Streamlit application for evaluating Large Language Models' ability to follow structured instructions in the context of an AI coding agent.

## Overview

This test suite evaluates multiple LLMs on their ability to:
- Parse and understand structured prompts
- Generate valid XML responses
- Follow security guidelines
- Handle ambiguous requests appropriately
- Execute complex command sequences

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenRouter API key in a `.env` file:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Running the Tests

Launch the Streamlit application:
```bash
streamlit run app.py
```

The UI provides:
- Test case status with visual indicators
- Real-time progress tracking
- Detailed results visualization
- LLM response comparison

## Test Cases

1. **Basic File Creation**
   - Tests simple file creation with specific content
   - Validates proper XML structure and path handling

2. **Composite Command Sequence**
   - Evaluates ability to chain multiple commands
   - Tests understanding of command dependencies

3. **Ambiguous Request Handling**
   - Tests clarification seeking behavior
   - Validates appropriate use of follow-up questions

4. **Nested File Operations**
   - Tests sequential file operations
   - Validates proper handling of file modes

5. **Path Validation Check**
   - Tests security awareness
   - Validates protection against path traversal

6. **Dangerous Command Prevention**
   - Tests command safety validation
   - Ensures protection against harmful operations

## Results Visualization

The application provides multiple views of test results:

1. **Success Rate Heatmap**
   - Shows success rates across models and test cases
   - Color-coded for quick visual analysis

2. **Response Time Analysis**
   - Compares average response times
   - Shows performance characteristics across models

3. **Detailed Results Table**
   - Comprehensive statistics for each model/test combination
   - Includes success rates and timing data

4. **LLM Responses**
   - Raw responses from each model
   - Validation status and error details
   - XML syntax highlighting

## Project Structure

```
instruct_following/
├── app.py                 # Streamlit UI application
├── test_llm_instruction_following.py  # Core testing logic
├── test_matrix.json      # Test case definitions
├── requirements.txt      # Project dependencies
└── README.md            # This documentation
```

## Models Tested

- deepseek/deepseek-chat
- anthropic/claude-3.5-sonnet
- anthropic/claude-3.5-haiku
- qwen/qwen-max

## Features

- Real-time test execution
- Configurable number of test passes
- Detailed validation of responses
- Security pattern checking
- Response time tracking
- Statistical analysis
- Comprehensive result visualization

## Implementation Details

The test suite uses:
- Streamlit for the UI
- OpenRouter API for LLM access
- Plotly for data visualization
- XML validation for response checking
- Async operations for parallel testing

## Results Analysis

Results are saved in multiple formats:
- `test_results.json`: Complete test data
- `test_results_detailed.csv`: Per-pass details
- `test_results_summary.csv`: Aggregated statistics
