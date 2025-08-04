# CodeInsights LLM Simulation

A comprehensive evaluation framework for assessing Large Language Models (LLMs) on C++ coding tasks across multiple simulation scenarios, focusing on functional correctness, student behavior alignment, and code quality metrics.

## Overview

This project evaluates LLMs' ability to:
1. Generate functionally correct C++ code
2. Mimic student coding patterns and behaviors
3. Replicate common programming mistakes
4. Match student efficiency and runtime characteristics

## Features

- **Multi-Scenario Evaluation**: 4 different scenarios testing various aspects of LLM coding capabilities
- **Multi-Model Support**: Supports both open-source and commercial LLMs
- **Comprehensive Metrics**: Functional correctness, code similarity, psychometric analysis, and efficiency alignment
- **Automated Testing**: Unit test execution and validation pipeline
- **Code Similarity Analysis**: AST edit distance and CodeBERT embeddings for semantic comparison

## Supported Models

### Commercial LLMs
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Gemini 2.5 Pro (Google)
- Mistral Large (Mistral AI)

### Open Source LLMs
- Google Gemma 3 27B Instruct
- Meta Llama 3.1 8B Instruct
- Qwen 2.5 14B Instruct

## Project Structure

```
├── compute_metrics.py           # Main metrics computation pipeline
├── run_commercialLLM_model.py   # Commercial LLM execution runner
├── utils.py                     # Core utilities and helper functions
├── cossim_calculator.py         # Code similarity calculation (AST + CodeBERT)
├── psychometrics_metrics.py     # Rasch model and psychometric analysis
└── codeinsights_llm_simulation/ # Data directory
    ├── scenario_results/        # Generated results by model
    ├── ability/                 # Student ability estimates
    ├── difficulty/              # Item difficulty estimates
    └── correlations/            # Correlation analysis results
```

## Evaluation Scenarios

### Scenario 1: Basic Functional Correctness
- **Goal**: Evaluate LLMs' ability to generate correct C++ solutions
- **Input**: Problem description and template
- **Metrics**: Functional correctness rate

### Scenario 2: Student Behavior Simulation
- **Goal**: Test LLMs' ability to mimic individual student coding styles
- **Input**: Student profile + coding examples + target problem
- **Metrics**: 
  - Functional correctness
  - AST edit distance from real student code
  - CodeBERT cosine similarity

### Scenario 3: Mistake Pattern Replication
- **Goal**: Evaluate LLMs' ability to reproduce common student mistakes
- **Input**: Student profile + mistake examples + target problem
- **Metrics**:
  - Unit test correctness alignment
  - Question-level mistake alignment (RMSE)
  - Code similarity metrics

### Scenario 4: Efficiency Alignment
- **Goal**: Test LLMs' ability to match student code efficiency
- **Input**: Student profile + efficiency examples + target problem
- **Metrics**:
  - Runtime efficiency alignment
  - All Scenario 2 & 3 metrics

## Installation

### Prerequisites
- Python 3.8+
- GCC compiler with C++17 support
- Required Python packages:

```bash
pip install pandas numpy scipy matplotlib seaborn
pip install transformers torch sklearn
pip install anthropic google-generativeai openai mistralai
pip install tree-sitter tree-sitter-cpp apted
pip install jinja2 requests
```

### API Keys Setup
Set the following environment variables:
```bash
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key" 
export OPENAI_API_KEY="your_openai_key"
export MISTRAL_API_KEY="your_mistral_key"
```

## Usage

### 1. Run Commercial LLM Evaluation
```bash
python run_commercialLLM_model.py
```
This script:
- Processes all 4 scenarios
- Runs evaluation with all commercial LLMs
- Saves results to `scenario_results/` directory
- Includes parallel processing with rate limiting

### 2. Compute Comprehensive Metrics
```bash
python compute_metrics.py
```
This script:
- Processes LLM outputs into structured DataFrames
- Executes generated code with unit tests
- Computes all evaluation metrics
- Saves results to `all_results.json`

### 3. Psychometric Analysis
```bash
python psychometrics_metrics.py
```
This script:
- Performs Rasch model analysis
- Computes ability and difficulty estimates
- Calculates correlations between models and ground truth
- Saves correlation results

### 4. Code Similarity Analysis
The `CodeSimilarityCalculator` class provides:
```python
from cossim_calculator import CodeSimilarityCalculator

calculator = CodeSimilarityCalculator()

# Calculate AST edit distance
distance = calculator.calculate_ast_edit_distance(code1, code2)

# Calculate CodeBERT similarity
similarity = calculator.calculate_codebert_similarity(code1, code2)

# Process entire DataFrame
result_df = calculator.process_dataframe(df)
```

## Key Metrics

### Functional Correctness
- **Definition**: Proportion of generated code that passes all unit tests
- **Range**: [0, 1] (higher is better)

### AST Edit Distance
- **Definition**: Normalized tree edit distance between ASTs
- **Range**: [0, 1] (lower indicates higher similarity)

### CodeBERT Cosine Similarity
- **Definition**: Semantic similarity using CodeBERT embeddings
- **Range**: [-1, 1] (higher indicates higher similarity)

### Unit Test Correctness Alignment
- **Definition**: Proportion of test cases where LLM and student have same pass/fail result
- **Range**: [0, 1] (higher is better)

### Question-Level Mistake Alignment
- **Definition**: RMSE of mistake proportions between LLM and students per question
- **Range**: [0, 1] (lower is better)

### Efficiency Alignment Score
- **Definition**: Average ratio of student runtime to LLM runtime
- **Interpretation**: Values near 1.0 indicate similar efficiency

## Data Format

### Input Data Structure
The system expects CSV files with the following key columns:
- `question_id`: Unique identifier for each coding problem
- `student_id`: Unique identifier for each student (Scenarios 2-4)
- `question_text`: Problem description
- `question_template`: C++ template with `{{ STUDENT_ANSWER }}` placeholder
- `question_unittests`: Formatted unit test cases
- `response`: Student's actual code solution

### Output Data Structure
Results are saved in JSON format with nested structure:
```json
{
  "S1": {
    "model_name": {
      "functional_correctness": 0.85,
      ...
    }
  },
  "S2": { ... },
  "S3": { ... },
  "S4": { ... }
}
```

## Technical Details

### Code Execution Pipeline
1. **Code Extraction**: Remove markdown fences and clean LLM output
2. **Template Rendering**: Insert code into Jinja2 templates with test cases
3. **Compilation**: Compile C++ code with GCC
4. **Execution**: Run with timeout protection (5 seconds)
5. **Validation**: Compare output against expected results

### Parallel Processing
- Commercial LLMs use ThreadPoolExecutor with rate limiting
- Configurable worker counts and delays per API
- Automatic retry and error handling

### Error Handling
- Compilation errors are logged and marked as failures
- Runtime errors and timeouts are handled gracefully
- Missing or malformed data is skipped with warnings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Submit a pull request

## License

This project is available under the MIT License. See LICENSE file for details.

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{codeinsights_llm_simulation,
  title={CodeInsights LLM Simulation: A Comprehensive Evaluation Framework for LLMs on Coding Tasks},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/your-repo/codeinsights-llm-simulation}}
}
```

## Troubleshooting

### Common Issues

1. **Compilation Errors**: Ensure GCC is installed with C++17 support
2. **API Rate Limits**: Adjust `delay` and `max_workers` parameters in LLM runners
3. **Memory Issues**: Large datasets may require increased system memory
4. **Missing Dependencies**: Install all required packages using pip

### Performance Optimization

- Use parallel processing for large datasets
- Cache intermediate results when possible
- Monitor API usage to avoid rate limits
- Consider using local models for development

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository.