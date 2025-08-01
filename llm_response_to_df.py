import requests
import pandas as pd

llms = [
    "google_gemma-3-27b-it",
    "meta_llama-3.1-8b-instruct",
    "qwen_qwen2.5-14b-instruct",
]
scenarios = ["S1", "S2", "S3", "S4"]

for llm in llms:
    for scenario in scenarios:
        path = f"https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/opensource_llm_output/{llm}/{scenario}/scenario_state.json"
        response = requests.get(path)
        raw = response.json()

        # Figure out where the state‐objects live
        if isinstance(raw, dict) and "request_states" in raw:
            states = raw["request_states"]
        elif isinstance(raw, list):
            states = raw
        else:
            raise ValueError(
                "JSON must be either a dict with 'request_states' or a list of state‐objects"
            )

        # Extract completion text
        texts = []
        for state in states:
            for comp in state.get("result", {}).get("completions", []):
                txt = comp.get("text")
                texts.append(txt)

        # Extract instance IDs
        ids = []
        for state in states:
            id_val = state.get("instance", {}).get("id")
            ids.append(str(id_val))

        # Create DataFrame
        if scenario == "S1":
            df = pd.DataFrame({"question_id": [int(x) for x in ids], "text": texts})
        else:
            parsed = [
                (int(sid), int(qid), text)
                for s, text in zip(ids, texts)
                if "_" in s
                for sid, qid in [s.split("_")]
            ]
            df = pd.DataFrame(parsed, columns=["student_id", "question_id", "text"])
