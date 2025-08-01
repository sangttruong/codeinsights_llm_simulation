import pandas as pd
import requests
import json
import re
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Your model configurations
MODELS = {
    "gemma-3-27b": {
        "base_url": "http://localhost:8611/v1/",
        "model_name": "google/gemma-3-27b-it"
    },
    "llama-3-8b": {
        "base_url": "http://localhost:8605/v1/",
        "model_name": "meta/llama-3-8b-chat"
    }
}

def load_dataset():
    """Load and process the dataset"""
    df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/Scenario1_full_data.csv")
    
    instances = []
    for question_id, question_df in df.groupby("question_unittest_id"):
        target = question_df.iloc[0]
        prompt = (
            f"Question: {target['question_name']} — {target['question_text']}\n\n"
            "Template:\n"
            f"{target['question_template']}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template. "
            "– Do NOT reproduce any part of the template "
            "– Do NOT emit int main() (it's already declared) "
            "– Ensure your code is correct, efficient, handles all edge cases, and includes any needed class definitions "
            "IMPORTANT: "
            "Your entire response must be exactly one Markdown C++ code-block. "
            "1. The first line of your output must be: "
            "```cpp "
            "2. The last line of your output must be: "
            "``` "
            "3. No extra characters, whitespace, or text may appear before the opening ```cpp or after the closing ```. "
            "Your output will therefore match this regex exactly: "
            "^```cpp\n([\\s\\S]+)\n```$"
        )
        
        instances.append({
            'question_id': question_id,
            'question_name': target['question_name'],
            'question_text': target['question_text'],
            'template': target['question_template'],
            'prompt': prompt
        })
    
    return instances

def call_vllm_model(base_url: str, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """Make a request to a vLLM model"""
    url = f"{base_url.rstrip('/')}/completions"
    
    # Default parameters optimized for code generation
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": kwargs.get("max_tokens", 1024),
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
        "stop": kwargs.get("stop", ["```\n\n", "```\n#", "\n\n\n"]),
        "echo": False,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

def extract_cpp_code(response_text: str) -> tuple[str, bool]:
    """Extract C++ code from the model response and check format validity"""
    # Primary pattern: exact format required
    pattern = r'^```cpp\n([\s\S]+)\n```$'
    match = re.match(pattern, response_text.strip())
    
    if match:
        return match.group(1), True
    
    # Fallback patterns for partially correct responses
    fallback_patterns = [
        r'```cpp\n([\s\S]+?)\n```',  # Standard cpp block
        r'```c\+\+\n([\s\S]+?)\n```',  # c++ variant
        r'```\n([\s\S]+?)\n```',  # Generic code block
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1), False
    
    return response_text.strip(), False

def run_single_instance(instance: Dict, model_name: str, model_config: Dict) -> Dict:
    """Run a single instance on a model"""
    start_time = time.time()
    
    result = {
        'question_id': instance['question_id'],
        'question_name': instance['question_name'],
        'model': model_name,
        'prompt': instance['prompt'],
        'response_text': '',
        'error': None,
        'response_time': 0
    }
    
    try:
        # Call the model
        response = call_vllm_model(
            model_config['base_url'], 
            model_config['model_name'], 
            instance['prompt']
        )
        
        if 'error' in response:
            result['error'] = response['error']
        else:
            # Extract the response text
            if 'choices' in response and len(response['choices']) > 0:
                response_text = response['choices'][0]['text']
                result['response_text'] = response_text
            else:
                result['error'] = "No choices in response"
    
    except Exception as e:
        result['error'] = str(e)
    
    result['response_time'] = time.time() - start_time
    return result

def run_evaluation(instances: List[Dict], max_workers: int = 4) -> Dict[str, List[Dict]]:
    """Run evaluation on all instances with all models"""
    results = {model_name: [] for model_name in MODELS.keys()}
    
    # Create all tasks
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_name, model_config in MODELS.items():
            for instance in instances:
                task = executor.submit(run_single_instance, instance, model_name, model_config)
                tasks.append((task, model_name))
        
        # Collect results
        completed_count = 0
        total_tasks = len(tasks)
        
        for task, model_name in tasks:
            try:
                result = task.result()
                results[model_name].append(result)
                completed_count += 1
                
                if completed_count % 10 == 0:
                    print(f"Completed {completed_count}/{total_tasks} tasks")
                    
            except Exception as e:
                print(f"Task failed: {e}")
                # Add error result
                results[model_name].append({
                    'question_id': 'unknown',
                    'model': model_name,
                    'error': str(e),
                })
    
    return results

def analyze_results(results: Dict[str, List[Dict]]) -> Dict:
    """Analyze and summarize the results"""
    analysis = {}
    
    for model_name, model_results in results.items():
        total = len(model_results)
        errors = sum(1 for r in model_results if r['error'] is not None)
        successful = total - errors
        
        avg_response_time = sum(r['response_time'] for r in model_results if r['response_time'] > 0) / max(successful, 1)
        
        analysis[model_name] = {
            'total_instances': total,
            'successful_responses': successful,
            'error_count': errors,
            'success_rate': successful / total if total > 0 else 0,
            'avg_response_time': avg_response_time
        }
    
    return analysis

def save_results(results: Dict, analysis: Dict, output_prefix: str = "cpp_evaluation"):
    """Save results to files"""
    # Save detailed results
    with open(f"{output_prefix}_detailed.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save analysis summary
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save CSV for easy analysis
    all_results = []
    for model_name, model_results in results.items():
        all_results.extend(model_results)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"{output_prefix}_results.csv", index=False)
    
    print(f"Results saved:")
    print(f"  - Detailed: {output_prefix}_detailed.json")
    print(f"  - Summary: {output_prefix}_summary.json")
    print(f"  - CSV: {output_prefix}_results.csv")

def print_summary(analysis: Dict):
    """Print a formatted summary of results"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, stats in analysis.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Total instances: {stats['total_instances']}")
        print(f"  Successful responses: {stats['successful_responses']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Average response time: {stats['avg_response_time']:.2f}s")
        print(f"  Errors: {stats['error_count']}")

def check_model_availability():
    """Check if vLLM models are available"""
    print("Checking model availability...")
    
    for model_name, config in MODELS.items():
        try:
            url = f"{config['base_url'].rstrip('/')}/models"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {model_name}: Available")
            else:
                print(f"✗ {model_name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"✗ {model_name}: {str(e)}")

def main():
    """Main execution function"""
    print("Starting C++ Code Generation Evaluation")
    
    # Check model availability
    check_model_availability()
    
    # Load dataset
    print("\nLoading dataset...")
    instances = load_dataset()
    print(f"Loaded {len(instances)} instances")


if __name__ == "__main__":
    instances = load_dataset()
    results  = run_evaluation(instances, max_workers=4)
    analysis = analyze_results(results)
    save_results(results, analysis, output_prefix="cpp_evaluation")
    print_summary(analysis)