import os
import json
import pandas as pd
from utils import (
    response_to_df,
    get_unittest_infos,
    compile_and_execute,
    compute_scenario_metrics,
    get_scenario_student_df,
)

LIST_LLMS = [
    "google_gemma-3-27b-it",
    "meta_llama-3.1-8b-instruct",
    "qwen_qwen2.5-14b-instruct",
]

LIST_SCENARIOS = ["S1", "S2", "S3", "S4"]

DATA_FOLDER = "my_dataset"
OUTPUT_FOLDER = "codeinsights_llm_simulation/"

if __name__ == "__main__":
    # Convert outputs to dataframe
    for scenario in LIST_SCENARIOS:
        for llm in LIST_LLMS:
            output_path = os.path.join(
                OUTPUT_FOLDER,
                f"opensource_llm_output/{llm}/{scenario}/scenario_state.json",
            )

            # Pre-processing model outputs
            output_df = response_to_df(path=output_path, scenario=scenario)

            # Save output dataframe
            scenario_id = scenario[1:]
            output_df.to_csv(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"scenario_results/{llm}/{llm}_scenario{scenario_id}.csv",
                )
            )

    # Read question data
    question_data = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "codeinsights_question.csv")
    )

    # Compute metrics
    all_results = {}
    for scenario in LIST_SCENARIOS:
        scenario_id = scenario[1:]
        scenario_student_df = get_scenario_student_df(
            scenario=scenario, data_folder=DATA_FOLDER
        )

        scenario_testcase_json = get_unittest_infos(
            scenario=scenario, data_folder=DATA_FOLDER
        )

        all_results[scenario] = {}

        for llm in LIST_LLMS:
            scenario_df = pd.read_csv(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"scenario_results/{llm}/{llm}_scenario{scenario_id}.csv",
                )
            )

            scenario_output_df = pd.read_csv(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"opensource_llm_output/{llm}/{scenario}/scenario_state.json",
                ),
                dtype={"question_id": str},
            )

            scenario_result_df = compile_and_execute(
                question_data=question_data,
                scenario_output_df=scenario_output_df,
                scenario_testcase_json=scenario_testcase_json,
            )

            all_results[scenario][llm] = compute_scenario_metrics(
                scenario=scenario,
                scenario_df=scenario_df,
                scenario_result_df=scenario_result_df,
                scenario_student_df=scenario_student_df,
            )

    # Save all results
    with open("all_results.json", "w", encoding="utf8") as f:
        json.dump(all_results, f, indent=2)
