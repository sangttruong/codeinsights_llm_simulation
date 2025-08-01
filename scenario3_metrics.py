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
import time
import ast
import numpy as np
from typing import Any, List, Tuple
import re

# For CodeBERT - you'll need to install: pip install transformers torch
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# For AST edit distance - you'll need to install: pip install apted tree-sitter tree-sitter-cpp
from apted import APTED
from apted.helpers import Tree
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

#Load Data
scenario3_df = pd.read_csv('/Users/kazunorifukuhara/Downloads/meta_llama-3.3-70b-instruct_scenario3.csv')
scenario3_df['question_id'] = scenario3_df['question_id'].astype(str)
scenario3_df['student_id'] = scenario3_df['student_id'].astype(str)
scenario3_student_df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/Scenario3_data.csv")
scenario3_student_df["student_id"] = scenario3_student_df["student_id"].astype(str)
scenario3_student_df["question_id"] = scenario3_student_df["question_id"].astype(int).astype(str)
question = pd.read_csv(
    '/Users/kazunorifukuhara/Downloads/Measurement from Dynamic Data Research/Important Data 4:16/final_question.csv'
)

#Define Function
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
#Class for CodeBERT and AST Edit Distance
class CodeSimilarityCalculator:
    def __init__(self):
        # Initialize CodeBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.eval()
        
        # Initialize tree-sitter for C++ parsing
        self.cpp_language = Language(tscpp.language())
        self.parser = Parser(self.cpp_language)
    
    def clean_code(self, code_str: str) -> str:
        """Clean code string by removing language markers and extra whitespace"""
        # Remove language markers like 'cpp\n' at the beginning
        code_str = re.sub(r'^(cpp|python|java|c\+\+)\s*\n', '', code_str, flags=re.IGNORECASE)
        # Remove extra whitespace
        return code_str.strip()
    
    def code_to_ast_tree(self, code: str) -> Tree:
        """Convert code to AST tree format for edit distance calculation"""
        try:
            # Clean the code
            clean_code = self.clean_code(code)            
            # Detect language and parse accordingly
            if self._is_cpp_code(clean_code):
                return self._cpp_to_tree(clean_code)
            else:
                try:
                    tree = ast.parse(clean_code)
                    return self._python_ast_to_tree(tree)
                except:
                    # If both fail, use a simple token-based tree
                    return self._tokenize_to_tree(clean_code)
        except Exception as e:
            print(f"Error parsing code: {e}")
            # Fallback: create a simple tree based on tokens
            return self._tokenize_to_tree(code)
    
    def _is_cpp_code(self, code: str) -> bool:
        """Detect if code is C++"""
        cpp_indicators = [
            '#include', 'std::', '::', 'cout', 'cin', 'endl',
            'class', 'public:', 'private:', 'protected:',
            'int main(', 'namespace', 'template<', 'vector<'
        ]
        return any(indicator in code for indicator in cpp_indicators)
    
    def _cpp_to_tree(self, code: str) -> Tree:
        """Convert C++ code to tree using tree-sitter"""
        try:
            # Parse with tree-sitter
            tree = self.parser.parse(bytes(code, 'utf8'))
            return self._simple_treesitter_to_apted(tree.root_node, depth=0)
        except Exception as e:
            print(f"Tree-sitter parsing failed: {e}")
            return self._tokenize_to_tree(code)
    
    def _simple_treesitter_to_apted(self, node, depth=0, max_depth=3) -> Tree:
        """Convert tree-sitter node to apted Tree with depth limiting"""
        try:
            # Limit recursion depth to prevent issues
            if depth > max_depth:
                return Tree.from_text("{leaf}")
            # Get node type safely
            node_type = str(getattr(node, 'type', 'unknown'))        
            # Clean node type to avoid parsing issues
            node_type = re.sub(r'[^a-zA-Z0-9_]', '_', node_type)   
            # For leaf nodes or when we're at max depth
            if not hasattr(node, 'children') or not node.children or depth >= max_depth:
                return Tree.from_text(f"{{{node_type}}}")   
            # Build tree string manually for APTED
            tree_parts = [node_type]
            child_count = 0     
            for child in node.children:
                if child_count >= 5:  # Limit children
                    break
                try:
                    child_tree = self._simple_treesitter_to_apted(child, depth + 1, max_depth)
                    if child_tree:
                        # Extract the tree representation
                        child_repr = str(child_tree).strip('{}')
                        tree_parts.append(child_repr)
                        child_count += 1
                except:
                    continue
            
            # Create tree string in APTED format
            if len(tree_parts) == 1:
                tree_str = f"{{{tree_parts[0]}}}"
            else:
                tree_str = "{" + "{".join(tree_parts) + "}" * len(tree_parts)
            
            return Tree.from_text(tree_str)
            
        except Exception as e:
            return Tree.from_text("{error}")
    
    def _treesitter_to_apted(self, node) -> Tree:
        """Convert tree-sitter node to apted Tree"""
        try:
            # Get node type - handle both string and node types
            if hasattr(node, 'type'):
                node_type = str(node.type)
            else:
                node_type = str(type(node).__name__)
            
            # Get children
            children = []
            if hasattr(node, 'children') and node.children:
                # Limit the number of children to prevent excessive tree size
                for child in node.children[:10]:  # Limit to first 10 children
                    try:
                        child_tree = self._treesitter_to_apted(child)
                        if child_tree:
                            children.append(child_tree)
                    except Exception as e:
                        # Skip problematic children
                        continue
            
            # Create tree - APTED expects specific format
            # Use Tree.from_text() for simple node creation
            if children:
                # For nodes with children, create a proper tree structure
                tree_str = f"{{{node_type}"
                for i, child in enumerate(children):
                    tree_str += f"{{{child.name if hasattr(child, 'name') else str(child)}}}"
                tree_str += "}"
                return Tree.from_text(tree_str)
            else:
                # For leaf nodes, try to include text content if available and short
                try:
                    if hasattr(node, 'text') and node.text:
                        text = node.text
                        if isinstance(text, bytes):
                            text = text.decode('utf-8', errors='ignore')
                        if len(str(text)) < 30:  # Limit text length
                            clean_text = re.sub(r'[{}]', '', str(text))  # Remove braces that confuse parser
                            return Tree.from_text(f"{{{node_type}:{clean_text}}}")
                except:
                    pass
                return Tree.from_text(f"{{{node_type}}}")
                
        except Exception as e:
            # Fallback to simple node representation
            return Tree.from_text(f"{{error_node_{hash(str(node)) % 1000}}}")
    
    def _tokenize_to_tree(self, code: str) -> Tree:
        """Create a simple token-based tree as fallback"""
        try:
            import re
            # Simple tokenization
            clean_code = self.clean_code(code)
            tokens = re.findall(r'\w+|[{}();,=+\-*/]', clean_code)
            
            # Limit tokens to prevent excessive computation
            tokens = tokens[:15]
            
            if not tokens:
                return Tree.from_text("{empty}")
            
            # Create a simple flat tree structure
            if len(tokens) == 1:
                return Tree.from_text(f"{{program{{{tokens[0]}}}}}")
            else:
                # Build nested structure manually
                tree_str = "{program"
                for token in tokens:
                    clean_token = re.sub(r'[^a-zA-Z0-9_]', '_', str(token))
                    tree_str += f"{{{clean_token}}}"
                tree_str += "}"
                return Tree.from_text(tree_str)
                
        except Exception as e:
            return Tree.from_text("{fallback}")
    
    def _python_ast_to_tree(self, node: Any) -> Tree:
        """Recursively convert Python AST node to Tree using from_text"""
        try:
            if isinstance(node, ast.AST):
                # Get node type
                node_type = type(node).__name__
                
                # Get child nodes
                children = []
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ast.AST):
                                children.append(self._python_ast_to_tree(item))
                    elif isinstance(value, ast.AST):
                        children.append(self._python_ast_to_tree(value))
                
                # Create tree string for APTED
                if children and len(children) <= 5:  # Limit children
                    tree_str = f"{{{node_type}"
                    for child in children:
                        child_str = str(child).strip('{}')
                        tree_str += f"{{{child_str}}}"
                    tree_str += "}"
                    return Tree.from_text(tree_str)
                else:
                    return Tree.from_text(f"{{{node_type}}}")
            else:
                # For non-AST nodes, convert to string
                clean_str = re.sub(r'[^a-zA-Z0-9_]', '_', str(node)[:20])
                return Tree.from_text(f"{{{clean_str}}}")
        except:
            return Tree.from_text("{error}")
    
    def calculate_ast_edit_distance(self, code1: str, code2: str) -> float:
        """Calculate AST edit distance between two code snippets"""
        try:
            # First try to create trees
            tree1 = self.code_to_ast_tree(code1)
            tree2 = self.code_to_ast_tree(code2)
            
            # Validate trees
            if not tree1 or not tree2:
                print("Failed to create valid trees, using fallback")
                return self._token_based_distance(code1, code2)
            
            # Calculate edit distance with error handling
            try:
                apted = APTED(tree1, tree2)
                distance = apted.compute_edit_distance()
            except Exception as apted_error:
                print(f"APTED calculation failed: {apted_error}")
                return self._token_based_distance(code1, code2)
            
            # Normalize the distance
            # Simple normalization based on average tree complexity
            code1_tokens = len(re.findall(r'\w+', self.clean_code(code1)))
            code2_tokens = len(re.findall(r'\w+', self.clean_code(code2)))
            avg_complexity = (code1_tokens + code2_tokens) / 2
            
            if avg_complexity == 0:
                return 0.0
            
            # Normalize distance (0 = identical, 1 = completely different)
            normalized_distance = min(1.0, distance / max(avg_complexity, 1))
            return float(normalized_distance)
            
        except Exception as e:
            print(f"Error calculating AST edit distance: {e}")
            # Fallback: use token-based comparison
            return self._token_based_distance(code1, code2)
    
    def _get_tree_size(self, tree) -> int:
        """Get approximate size of tree"""
        try:
            if not hasattr(tree, 'children') or not tree.children:
                return 1
            
            total_size = 1  # Count current node
            for child in tree.children:
                total_size += self._get_tree_size(child)
                if total_size > 100:  # Prevent excessive computation
                    return 100
            return total_size
        except:
            return 1
    
    def _token_based_distance(self, code1: str, code2: str) -> float:
        """Fallback token-based distance calculation"""
        try:
            # Clean and tokenize both codes
            clean1 = self.clean_code(code1)
            clean2 = self.clean_code(code2)
            
            # Simple tokenization
            tokens1 = re.findall(r'\w+|[{}();,=+\-*/]', clean1)
            tokens2 = re.findall(r'\w+|[{}();,=+\-*/]', clean2)
            
            if not tokens1 and not tokens2:
                return 0.0
            if not tokens1 or not tokens2:
                return 1.0
            
            # Calculate Jaccard distance
            set1 = set(tokens1)
            set2 = set(tokens2)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = intersection / union
            return 1.0 - jaccard_similarity
            
        except Exception as e:
            print(f"Fallback distance calculation failed: {e}")
            return 1.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance as fallback"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_code_embedding(self, code: str) -> np.ndarray:
        """Get CodeBERT embedding for code snippet"""
        try:
            # Clean the code
            clean_code = self.clean_code(code)
            
            # Tokenize
            tokens = self.tokenizer(
                clean_code, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**tokens)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.numpy().flatten()
        except Exception as e:
            print(f"Error getting CodeBERT embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(768)  # CodeBERT embedding dimension
    
    def calculate_codebert_similarity(self, code1: str, code2: str) -> float:
        """Calculate CodeBERT cosine similarity between two code snippets"""
        try:
            emb1 = self.get_code_embedding(code1)
            emb2 = self.get_code_embedding(code2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating CodeBERT similarity: {e}")
            return 0.0
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe to add similarity metrics"""
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Initialize lists to store results
        ast_distances = []
        codebert_similarities = []
        
        for idx, row in df.iterrows():
            llm_response = str(row['LLM_response'])
            student_response = str(row['student_response'])
            
            # Calculate AST edit distance
            ast_dist = self.calculate_ast_edit_distance(llm_response, student_response)
            ast_distances.append(ast_dist)
            
            # Calculate CodeBERT similarity
            codebert_sim = self.calculate_codebert_similarity(llm_response, student_response)
            codebert_similarities.append(codebert_sim)
        
        # Add new columns to dataframe
        result_df['ast_edit_distance'] = ast_distances
        result_df['codebert_cosine_similarity'] = codebert_similarities
        
        return result_df

#Prepare test case for scenario2
scenario3_student_df['tests'] = scenario3_student_df['question_unittests'].apply(parse_unittests)
result = { str(int(qid)): tests for qid, tests in zip(scenario3_student_df['question_id'], scenario3_student_df['tests']) }
jsonfile = json.dumps(result, indent=2)
scenario3_testcase_json = json.loads(jsonfile)

#Run LLM generated codes and make result dataframe
scenario3_results = []

for i in range(len(scenario3_df)):
    test_result = extract_student_code(scenario3_df.iloc[i]["text"])
    id_val = int(scenario3_df.iloc[i]["question_id"])
    student_id = scenario3_df.iloc[i]["student_id"]
    sub_question = question[question["question_id"] == id_val]

    # format testcases
    try:
        formatted_testcases, std_inputs = format_testcases(scenario3_testcase_json[str(id_val)])
        expected_output = [tc["expected_output"] for tc in formatted_testcases]
    except KeyError:
        print(f"Question ID {id_val} not found in test_case dictionary.")
        # record the failure with no specific testcase
        scenario3_results.append({
            "student_id": student_id,
            "question_id": id_val,
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
        print(f"Error generating code for question ID {id_val}: {e}", file=sys.stderr)
        scenario3_results.append({
            "student_id": student_id,
            "question_id": id_val,
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
            "student_id": student_id,
            "question_id": id_val,
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

            # 1) Compile
            compile_proc = subprocess.run(
                ["g++", "-std=c++17", cpp_path, "-o", exe_path],
                capture_output=True, text=True
            )
            if compile_proc.returncode != 0:
                # record failure, leave stdout as None (or store compile_proc.stderr if you like)
                scenario3_results.append(row)
                continue

            # 2) Run
            try:
                run_proc = subprocess.run(
                    [exe_path],
                    capture_output=True,
                    text=True,             # keep automatic newline translation
                    encoding='utf-8',      # explicitly say “use UTF‑8”
                    errors='replace',      # replace any invalid byte with �
                    timeout=5
                )
            except subprocess.TimeoutExpired as e:
                print(f"Testcase {j} for question {id_val} timed out after {e.timeout}s", file=sys.stderr)
                # record the timeout as a failure
                scenario3_results.append(row)
                continue

            if run_proc.returncode != 0:
                print("Runtime error:\n", run_proc.stderr, file=sys.stderr)
                scenario3_results.append(row)
                continue

            # success → capture output
            row["stdout"] = run_proc.stdout
            scenario3_results.append(row)
scenario3_result_df = pd.DataFrame(scenario3_results)
scenario3_result_df["correctness"] = pd.Series(dtype="Int64")

# Where stdout == expected_output → 1, else → 0
scenario3_result_df["stdout"] = scenario3_result_df["stdout"].apply(
    lambda x: x.strip() if isinstance(x, str) else x
)
matches = scenario3_result_df["stdout"] == scenario3_result_df["expected_output"]
scenario3_result_df.loc[matches, "correctness"] = 1
scenario3_result_df.loc[~matches, "correctness"]     = 0

# If test_case_id is missing, reset correctness to <NA>
#    (this handles both None and np.nan)
mask_missing = scenario3_result_df["test_case_id"].isna()
scenario3_result_df.loc[mask_missing, "correctness"] = pd.NA
scenario3_result_df['question_id'] = scenario3_result_df['question_id'] \
    .astype(str)
scenario3_result_df['test_case_id'] = scenario3_result_df['test_case_id'] \
    .astype(str)
scenario3_result_df = scenario3_result_df.rename(columns={
    "correctness": "LLM_correctness",
})
functional_correctnes = scenario3_result_df["LLM_correctness"].mean()
print(f"Functional correctness: {functional_correctnes}")

#Unit test correctness alignment
#Student Pass Patterns
scenario3_student_df['num_unittest'] = scenario3_student_df['question_unittests'] \
    .astype(str) \
    .str.count(r'Unittest\s+\d+:')
scenario3_student_df['pass_str'] = scenario3_student_df['pass'].astype(str).str.split('.', n=1).str[0]
# 2) pad on the right with “0” to match num_unittest
scenario3_student_df['pass_padded'] = scenario3_student_df.apply(
    lambda r: r['pass_str'].ljust(int(r['num_unittest']), '0'),
    axis=1
)
# 3) explode into one digit per row
df_exploded = (
    scenario3_student_df
    .assign(pass_list = scenario3_student_df['pass_padded'].apply(lambda s: [int(ch) for ch in s]))
    .explode('pass_list')
    .reset_index()    # keep original row‑index for grouping
)

# 4) get your unittest_id and rename columns
df_result = (
    df_exploded
    .assign(unittest_id = df_exploded.groupby('index').cumcount())
    .loc[:, ["student_id",'question_id', 'unittest_id', 'pass_list']]
    .rename(columns={'pass_list':'pass_unittest'})
    .reset_index(drop=True)
)
df_result['question_id'] = df_result['question_id'] \
    .astype(int) \
    .astype(str)
df_result['student_id'] = df_result['student_id'] \
    .astype(str)
df_result['unittest_id'] = df_result['unittest_id'] \
    .astype(str)
df_result = df_result.rename(columns={
    "unittest_id": "test_case_id",
    "pass_unittest": "real_student_correctness",
})
scenario3_merged_df = scenario3_result_df.merge(
    df_result,
    on=["student_id", "question_id", "test_case_id"],
    how="inner",
    suffixes=("_sc2", "_sc1")    # optional, to disambiguate any overlapping column names
)
mask = (
    scenario3_merged_df['LLM_correctness'].isin([0, 1]) &
    scenario3_merged_df['real_student_correctness'].isin([0, 1])
)
valid = scenario3_merged_df[mask]

# 2) check equality
matches = valid['LLM_correctness'] == valid['real_student_correctness']
unit_test_correctness_alignment = matches.mean()
print(f"Unit Test Correctness Alignment: {unit_test_correctness_alignment:.2%}")

#ASTED and CodeBERT Similarity
student_response_subset = scenario3_student_df[["student_id", "question_id", "response"]]
student_response_subset = student_response_subset.rename(columns={
     "response": "student_response"
 })
scenario3_df = scenario3_df.rename(columns={
    "text": "LLM_response"
})
scenario3_response_full = scenario3_df.merge(
    student_response_subset,
    on=["student_id", "question_id"],
    how="inner"
)
calculator = CodeSimilarityCalculator()
result_df = calculator.process_dataframe(scenario3_response_full)
ast_edit_distance = result_df['ast_edit_distance'].mean()
codebert_cosine_similarity = result_df['codebert_cosine_similarity'].mean()
print(f"AST Edit Distance: {ast_edit_distance}")
print(f"CodeBERT Cosine Similarity: {codebert_cosine_similarity}")

#Question-level mistake alignment
question_props = (
    scenario3_merged_df
    .groupby('question_id')
    .agg(
        LLM_question_mistake_prop       = ('LLM_correctness',       lambda x: (x == 0).mean()),
        student_question_mistake_prop   = ('real_student_correctness', lambda x: (x == 0).mean())
    )
    .reset_index()
)

# 2) Compute squared difference per question
question_props['squared_diff'] = (
    question_props['LLM_question_mistake_prop']
  - question_props['student_question_mistake_prop']
) ** 2

# 3) RMSE across all questions
question_level_mistake_alignment_score = np.sqrt(question_props['squared_diff'].mean())
print("Question‑level mistake alignment score:", question_level_mistake_alignment_score)
