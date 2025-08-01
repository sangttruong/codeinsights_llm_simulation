import pandas as pd
import re
import os
import subprocess
import tempfile
import shutil
import requests
from jinja2 import Template
from typing import List, Dict
import sys
import pprint
import json

#Load Data
#Load LLM generated code
llm = "mistral"
scenario = 1
scenario1_df = pd.read_csv('https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/scenario_results/{llm}/{llm}_scenario{scenario}.csv')
scenario1_df['question_id'] = scenario1_df['question_id'].astype(str)
scenario1_student_df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/Scenario1_2_data.csv")
question = pd.read_csv(
    'https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/codeinsights_question.csv'
)

#Define Functions
def extract_student_code(model_code: str) -> str:
    """
    Extracts clean C++ code from model output:
    - Trims preambles
    - Removes student's main()
    """
    code_blocks = re.findall(r"```(?:c\+\+)?\n(.*?)```", model_code, flags=re.DOTALL)
    if code_blocks:
        model_code = code_blocks[0].strip()  # Use the first code block
        print("[Markdown extraction] Used fenced code blocks.")

    # Post-processing
    # Comment out as a testing - 7/3/2025
    lines = model_code.strip().splitlines()
    start_keywords = ("#include", "using namespace")
    for i, line in enumerate(lines):
        if any(line.strip().startswith(k) for k in start_keywords):
            lines[i] = ""
    code = "\n".join(lines).strip()
    if "int main" in code:
        code = code.split("int main")[0].strip()

    # --- Final touch ---
    if "print(" in code and "void print()" not in code and "print()" not in code:
        print("⚠️ WARNING: `print()` is called in test input but not defined.")

    return code

def format_testcases(test_case):
        """Formats the test cases into the required format for the grading engine.

        Returns:
            Tuple[List[Dict[str]], List[str]]: A tuple containing the formatted test cases and standard inputs.
        """
        formatted_testcases = []
        std_inputs = []
        for testcase in test_case:
            formatted_testcases.append(
                {
                    "testcode": testcase["input"],
                    "expected_output": testcase["output"],
                }
            )
            if "std_in" not in testcase:
                std_inputs.append("")
            else:
                std_inputs.append(testcase["std_in"])
        return formatted_testcases, std_inputs

def generate_code(
    template: str,
    student_answer: str,
    formatted_testcases: List[Dict[str, str]],
) -> List[str]:
    """
    Generates one C++ file per test case by rendering the Jinja2 template.

    Args:
        template:     Your question_template (with Jinja2 tags).
        student_answer:  The raw LLM output, including ```cpp fences.
        formatted_testcases: A list of dicts, where each dict provides the keys
            'extra' (pre‑test code) and 'testcode' (the call or check).

    Returns:
        A list of fully rendered C++ source strings, one per testcase.
    """
    # Strip any ```cpp ... ``` markdown fences
    student_answer = re.sub(r"^```cpp\s*|\s*```$", "", student_answer)

    # Compile the Jinja2 template once
    j2 = Template(template)

    rendered_codes: List[str] = []
    error_flags:    List[bool] = []

    rendered_codes: List[str] = []
    for tc in formatted_testcases:
        # clean up the testcase
        tc["testcode"] = tc["testcode"].replace("STD input:", "").strip()

        try:
            code = j2.render(
                STUDENT_ANSWER=student_answer,
                TESTCASES=[tc]
            )
        except TypeError as e:
            # record that this testcase failed, log if you like
            error_flags.append(True)
            print(f"[Warning] TypeError rendering testcase {tc!r}: {e}", file=sys.stderr)
            continue
        else:
            error_flags.append(False)
            rendered_codes.append(code)

    return rendered_codes
def parse_unittests(block: str):
    pattern = re.compile(
        r'Unittest\s*(\d+):\s*'
        r'Input:\s*(.*?)\s*'
        r'Output:\s*(.*?)(?=Unittest\s*\d+:|$)',
        flags=re.DOTALL
    )
    tests = []
    for m in pattern.finditer(block):
        tests.append({
            "unittest":   m.group(1).strip(),
            "input":      m.group(2).strip(),
            "output":     m.group(3).strip()
        })
    return tests

scenario1_student_df['tests'] = scenario1_student_df['question_unittests'].apply(parse_unittests)
result = { str(int(qid)): tests for qid, tests in zip(scenario1_student_df['question_id'], scenario1_student_df['tests']) }
jsonfile = json.dumps(result, indent=2)
scenario1_testcase_json = json.loads(jsonfile)

#Run LLM generated codes and make result dataframe
scenario1_results = []

for i in range(len(scenario1_df)):
    test_result = extract_student_code(scenario1_df.iloc[i]["text"])
    id_val = int(scenario1_df.iloc[i]["question_id"])
    sub_question = question[question["question_id"] == id_val]
    print(id_val)
    # format testcases
    try:
        formatted_testcases, std_inputs = format_testcases(scenario1_testcase_json[str(id_val)])
        expected_output = [tc["expected_output"] for tc in formatted_testcases]
    except KeyError:
        # record the failure with no specific testcase
        scenario1_results.append({
            "id": id_val,
            "test_case_id": None,
            "cpp_code": None,
            "stdout": None,
            "expected_output": None
        })
        continue
    # generate code
    try:
        codes = generate_code(
            sub_question["question_template"].iloc[0],
            test_result,
            formatted_testcases
        )
    except Exception as e:
        scenario1_results.append({
            "id": id_val,
            "test_case_id": None,
            "cpp_code": None,
            "stdout": None,
            "expected_output": None
        })
        continue

    # compile & run each snippet
    for j, cpp_code in enumerate(codes):
        stdout_val = None
        expected_output_val = expected_output[j]
        # default row template
        row = {
            "id": id_val,
            "test_case_id": j,
            "cpp_code": cpp_code,
            "stdout": None,
            "expected_output": expected_output_val
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "test.cpp")
            exe_path = os.path.join(tmpdir, "test")

            # write out the code
            with open(cpp_path, "w") as f:
                f.write(cpp_code)

            # Compile
            compile_proc = subprocess.run(
                ["g++", "-std=c++17", cpp_path, "-o", exe_path],
                capture_output=True, text=True
            )
            if compile_proc.returncode != 0:
                # record failure, leave stdout as None (or store compile_proc.stderr if you like)
                scenario1_results.append(row)
                continue

            try:
                run_proc = subprocess.run(
                    [exe_path],
                    capture_output=True,
                    text=True,
                    timeout=5          # ← kill the process if it runs longer than 10s
                )
            except subprocess.TimeoutExpired as e:
                print(f"Testcase {j} for question {id_val} timed out after {e.timeout}s", file=sys.stderr)
                # record the timeout as a failure
                scenario1_results.append(row)
                continue

            if run_proc.returncode != 0:
                print("Runtime error:\n", run_proc.stderr, file=sys.stderr)
                scenario1_results.append(row)
                continue
            row["stdout"] = run_proc.stdout
            scenario1_results.append(row)
#Store result in dataframe
scenario1_result_df = pd.DataFrame(scenario1_results)

#Evaluate correctness of LLM generated code
scenario1_result_df["correctness"] = pd.Series(dtype="Int64")
# Where stdout == expected_output → 1, else → 0
matches = scenario1_result_df["stdout"] == scenario1_result_df["expected_output"]
scenario1_result_df.loc[matches, "correctness"] = 1
scenario1_result_df.loc[~matches, "correctness"]     = 0
# If test_case_id is missing, reset correctness to <NA>
mask_missing = scenario1_result_df["test_case_id"].isna()
scenario1_result_df.loc[mask_missing, "correctness"] = pd.NA
scenario1_result_df["correctness"] = pd.Series(dtype="Int64")
# Where stdout == expected_output → 1, else → 0
scenario1_result_df["stdout"] = scenario1_result_df["stdout"].apply(
    lambda x: x.strip() if isinstance(x, str) else x
)
scenario1_result_df = scenario1_result_df.rename(columns={
    "correctness": "LLM_correctness",
})
functional_correctnes = scenario1_result_df["LLM_correctness"].mean()
print(f"Functional correctness: {functional_correctnes:.2%}")
