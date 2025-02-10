import streamlit as st
import json
import pandas as pd
import plotly.express as px
import asyncio
from pathlib import Path
from test_llm_instruction_following import ToolCallTester

st.set_page_config(
    page_title="LLM Testing Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_test_matrix():
    with open("test_matrix.json", "r") as f:
        return json.load(f)

def format_test_description(test):
    desc = test["description"]
    if "expected_elements" in test:
        desc += "\n(Expects specific XML elements)"
    elif "expected_pattern" in test:
        patterns = [p["element"] for p in test["expected_pattern"]]
        desc += f"\n(Expects: {', '.join(patterns)})"
    return desc

# Custom CSS for better styling
st.markdown("""
    <style>
    .stProgress .st-bo {
        background-color: #e0e0e0;
    }
    .success {
        color: #28a745;
    }
    .warning {
        color: #ffc107;
    }
    .danger {
        color: #dc3545;
    }
    .test-matrix {
        position: sticky;
        top: 0;
        background: white;
        z-index: 999;
        padding: 10px 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    .badge-inprog {
        background-color: #1e90ff;
    }
    .badge-done {
        background-color: #28a745;
    }
    .badge-waiting {
        background-color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

def display_test_matrix(matrix, running_test=None, completed_tests=None):
    """Display the test matrix with status indicators."""
    completed_tests = completed_tests or set()
    
    # Create placeholders for each test case if they don't exist
    if "test_placeholders" not in st.session_state:
        st.session_state.test_placeholders = []
        for i in range(len(matrix)):
            ph = st.empty()
            st.session_state.test_placeholders.append(ph)
    
    # Update each test case's display
    for i, test in enumerate(matrix):
        ph = st.session_state.test_placeholders[i]
        
        with ph.container():
            cols = st.columns([0.15, 0.85])
            
            # Status badge
            with cols[0]:
                if i in completed_tests:
                    st.markdown('<span class="badge badge-done">DONE</span>', unsafe_allow_html=True)
                elif running_test == i:
                    st.markdown('<span class="badge badge-inprog">IN-PROG</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-waiting">WAITING</span>', unsafe_allow_html=True)
            
            # Test details
            with cols[1]:
                with st.expander(test['description']):
                    st.text(f"Prompt: {test['prompt']}")
                    if "expected_elements" in test:
                        st.text("Expected Elements:")
                        for elem in test["expected_elements"]:
                            st.code(f"<{elem[0]} {json.dumps(elem[1])}>{elem[2]}</{elem[0]}>")
                    if "expected_pattern" in test:
                        st.text("Expected Patterns:")
                        for pattern in test["expected_pattern"]:
                            st.code(f"{pattern['element']}: {pattern.get('content_contains', '')}")

def display_models():
    models = [
        "deepseek/deepseek-chat",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku",
        "qwen/qwen-max"
    ]
    st.subheader("Models Being Tested")
    for model in models:
        st.code(model)

def create_results_dataframe(results):
    data = []
    for model in results["results"]:
        for test_idx, test_data in results["results"][model].items():
            stats = test_data.get("statistics", {})
            row = {
                "Model": model,
                "Test": test_data["description"],
                "Success Rate": stats.get("success_rate", 0.0),
                "Avg Response Time": stats.get("avg_response_time", 0.0),
                "Response Time Stdev": stats.get("response_time_stdev", 0.0),
                "Min Response Time": stats.get("min_response_time", 0.0),
                "Max Response Time": stats.get("max_response_time", 0.0)
            }
            data.append(row)
    return pd.DataFrame(data)

def display_llm_responses(results):
    """Display detailed LLM responses with validation status."""
    st.subheader("LLM Responses")
    
    # Group by test first, then model
    test_indices = sorted(set(idx for model in results["results"].values() for idx in model.keys()))
    for test_idx in test_indices:
        # Get test description from any model
        test_desc = next(iter(results["results"].values()))[test_idx]["description"]
        with st.expander(f"Test: {test_desc}", expanded=False):
            for model in results["results"]:
                st.markdown(f"**Model: {model}**")
                test_data = results["results"][model][test_idx]
                for i, pass_data in enumerate(test_data["passes"], 1):
                    with st.container():
                        cols = st.columns([0.15, 0.85])
                        with cols[0]:
                            if pass_data.get("error"):
                                st.error(f"Pass {i}")
                            elif pass_data.get("validation", {}).get("valid_xml", False):
                                st.success(f"Pass {i}")
                            else:
                                st.warning(f"Pass {i}")
                        
                        with cols[1]:
                            if pass_data.get("error"):
                                st.error(pass_data["error"])
                            else:
                                validation = pass_data.get("validation", {})
                                if not validation.get("valid_xml", False):
                                    if validation.get("element_errors"):
                                        st.error(f"Element errors: {', '.join(validation['element_errors'])}")
                                    if validation.get("security_issues"):
                                        st.error(f"Security issues: {', '.join(validation['security_issues'])}")
                                st.code(pass_data["response"], language="xml")
                st.markdown("---")

def display_results(results_df, raw_results=None):
    st.subheader("Test Results")
    
    # Success rate heatmap
    pivot_success = results_df.pivot(index="Test", columns="Model", values="Success Rate")
    fig_success = px.imshow(
        pivot_success,
        labels=dict(x="Model", y="Test", color="Success Rate"),
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    fig_success.update_layout(
        title="Success Rate by Model and Test",
        height=400
    )
    st.plotly_chart(fig_success, use_container_width=True)
    
    # Response time comparison
    fig_time = px.bar(
        results_df,
        x="Model",
        y="Avg Response Time",
        color="Test",
        title="Average Response Time by Model and Test",
        barmode="group"
    )
    fig_time.update_layout(height=400)
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    # Format the dataframe without gradient
    results_df = results_df.round(3)
    results_df['Success Rate'] = results_df['Success Rate'].map('{:.1%}'.format)
    results_df['Avg Response Time'] = results_df['Avg Response Time'].map('{:.2f}s'.format)
    results_df['Response Time Stdev'] = results_df['Response Time Stdev'].map('{:.2f}s'.format)
    st.dataframe(results_df, use_container_width=True)
    

async def run_tests(tester=None):
    completed_tests = set()
    current_test = None
    progress_placeholder = st.empty()
    
    try:
        if tester is None:
            tester = ToolCallTester(passes=1)  # Default to 1 pass if not specified
        
        def progress_callback(desc):
            nonlocal current_test
            
            # Extract test number from description
            if desc.startswith("Test"):
                test_num = int(desc.split("/")[0].split(" ")[1]) - 1
                if test_num != current_test:
                    if current_test is not None:
                        completed_tests.add(current_test)
                    current_test = test_num
                
                # Update progress message
                progress_placeholder.markdown(f"**Running:** {desc}")
                
                # Update test matrix display
                display_test_matrix(tester.test_matrix, current_test, completed_tests)
        
        results = await tester.run_all_tests(progress_callback)
        
        # Mark the last test as completed
        if current_test is not None:
            completed_tests.add(current_test)
            display_test_matrix(tester.test_matrix, None, completed_tests)
        
        progress_placeholder.empty()
        st.success("Testing Complete!")
        return results
    except Exception as e:
        st.error(f"Error during testing: {str(e)}")
        st.error("Test run failed. Please check the error message above.")
        return None

def main():
    st.title("LLM Instruction Following Test Suite")
    
    try:
        # Initialize session state
        if "running" not in st.session_state:
            st.session_state.running = False
            st.session_state.results = None
        
        # Load test matrix
        matrix = load_test_matrix()
        
        # Create main layout
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            # Test matrix section
            test_container = st.container()
            with test_container:
                st.subheader("Test Cases")
                display_test_matrix(matrix)
            
            # Results section
            if "results_placeholder" not in st.session_state:
                st.session_state.results_placeholder = st.empty()
            
            if st.session_state.results:
                with st.session_state.results_placeholder.container():
                    results_df = create_results_dataframe(st.session_state.results)
                    display_results(results_df, st.session_state.results)
                    display_llm_responses(st.session_state.results)
        
        with col2:
            # Controls section
            control_container = st.container()
            with control_container:
                num_passes = st.number_input("Number of passes", min_value=1, max_value=5, value=1)
                st.button("Run Tests", type="primary", 
                         disabled=st.session_state.running,
                         on_click=lambda: setattr(st.session_state, 'running', True))
            
            st.markdown("---")
            display_models()
        
        # Handle test execution
        if st.session_state.running:
            try:
                tester = ToolCallTester(passes=num_passes)
                results = asyncio.run(run_tests(tester=tester))
                if results:
                    st.session_state.results = results
                    # Update results immediately
                    with st.session_state.results_placeholder.container():
                        results_df = create_results_dataframe(results)
                        display_results(results_df, results)
                        display_llm_responses(results)
            finally:
                st.session_state.running = False
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")

if __name__ == "__main__":
    main()
