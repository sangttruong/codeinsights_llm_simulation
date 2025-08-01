import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from tueplots import bundles

df1 = pd.read_csv('/Users/kazunorifukuhara/Downloads/Model Response Results/gemma-3-27b-it_scenario2_result.csv')
df2 = pd.read_csv('/Users/kazunorifukuhara/Downloads/Model Response Results/gemma-3-27b-it_scenario3_result.csv')
df3 = pd.read_csv('/Users/kazunorifukuhara/Downloads/Model Response Results/gemma-3-27b-it_scenario4_result.csv')
df = pd.concat(
    [df1, df2, df3],
    axis=0,           # stack rows
    ignore_index

df["item_id"] = (df["question_id"].astype(str) + "_" + df["test_case_id"].astype(str))

K = 0.4
initial_rating = 0.0
# 2) Lazy init for any new student_id or item_id
def get_rating(store, key):
    return store.setdefault(key, initial_rating)

# 3) Elo update with clipping
def rasch_update(theta, z, resp, K=0.4):
    """
    One‐step update using a Rasch‐style logistic for p_{ij}:
      p = 1 / (1 + exp[-(theta - z)])
    then delta = K*(resp - p), and clip back into [min_r, max_r].
    
    Args:
      theta   float: current ability (θ_i)
      z       float: current difficulty (z_j)
      resp    {0,1}: observed response R_{ij}
      K       float: learning rate
      min_r   float: lower bound for both θ_i and z_j
      max_r   float: upper bound for both θ_i and z_j
    Returns:
      (theta_new, z_new)
    """
    # 1) Rasch‐style probability
    p = 1.0 / (1.0 + np.exp(-(theta - z)))
    
    # 2) compute update magnitude
    delta = K * (resp - p)
    
    # 3) apply updates
    theta_new = theta + delta
    z_new     = z     - delta
    
    return theta_new, z_new

# 4) Run through your data in order
student_ratings = {}
item_ratings    = {}

for _, row in df.iterrows():
    sid, iid, resp = (
        row["student_id"],
        row["item_id"],
        row["LLM_correctness"],  # assumed 0 or 1
    )
    R_s = get_rating(student_ratings, sid)
    R_i = get_rating(item_ratings,    iid)

    R_s_new, R_i_new = rasch_update(
        R_s, R_i, resp,
        K=K
    )

    student_ratings[sid] = R_s_new
    item_ratings[iid]    = R_i_new

students_df = (
    pd.DataFrame.from_dict(student_ratings, orient="index", columns=["ability"])
      .reset_index().rename(columns={"index": "student_id"})
)
items_df = (
    pd.DataFrame.from_dict(item_ratings, orient="index", columns=["difficulty"])
      .reset_index().rename(columns={"index": "item_id"})
)

def load_and_prefix(path, model_name):
    # only read the columns we need
    df = pd.read_csv(path, usecols=["student_id", "ability"])
    # rename ability → ability_<model_name>
    df = df.rename(columns={"ability": f"ability_{model_name}"})
    return df

# 2) load each one
claude_df  = load_and_prefix("CodeInsights LLM Abilities/claude-3-5_student_ability.csv",  "claude")
mistral_df = load_and_prefix("CodeInsights LLM Abilities/mistral_student_ability.csv",   "mistral")
gemini_df  = load_and_prefix("CodeInsights LLM Abilities/gemini-2.5-pro_student_ability.csv", "gemini")
gpt_df     = load_and_prefix("CodeInsights LLM Abilities/gpt-4o_student_ability.csv",     "gpt4o")
llama_df   = load_and_prefix("CodeInsights LLM Abilities/llama3.1-8b_student_ability.csv","llama3.1")
gemma_df  = load_and_prefix("CodeInsights LLM Abilities/gemma-3-27b-it_student_ability.csv",  "gemma")
real_df    = load_and_prefix("CodeInsights LLM Abilities/real_student_ability.csv",  "real")
merged = reduce(
    lambda left, right: pd.merge(left, right, on="student_id", how="outer"),
    [claude_df, mistral_df, gemini_df, gpt_df, llama_df, gemma_df, real_df]
)
# 1) identify the comparison columns
ability_cols = [c for c in merged.columns 
                if c.startswith("ability_") and c != "ability_real"]

# 2) compute correlations + p‑values one‑by‑one
results = []
for col in ability_cols:
    # drop rows where either real or this model is NaN
    sub = merged[["ability_real", col]].dropna()

    # Pearson
    r, p_r = pearsonr(sub["ability_real"], sub[col])
    # Spearman
    s, p_s = spearmanr(sub["ability_real"], sub[col])

    results.append({
        "model":       col.replace("ability_", ""),
        "ability_corr":   r,
    })

# 3) assemble into a DataFrame
ability_corr_df = pd.DataFrame(results)

#Difficulty
def load_and_prefix(path, model_name):
    # only read the columns we need
    df = pd.read_csv(path, usecols=["item_id", "difficulty"])
    # rename ability → ability_<model_name>
    df = df.rename(columns={"difficulty": f"difficulty_{model_name}"})
    return df

# 2) load each one
claude_difficulty  = load_and_prefix("CodeInsights_LLM_Difficulties/mistral_item_difficulty.csv",  "claude")
mistral_difficulty = load_and_prefix("CodeInsights_LLM_Difficulties/mistral_item_difficulty.csv",   "mistral")
gemini_difficulty  = load_and_prefix("CodeInsights_LLM_Difficulties/gemini-2.5-pro_item_difficulty.csv", "gemini")
gpt_difficulty     = load_and_prefix("CodeInsights_LLM_Difficulties/gpt-4o_item_difficulty.csv",     "gpt4o")
llama_difficulty   = load_and_prefix("CodeInsights_LLM_Difficulties/llama3.1-8b_item_difficulty.csv","llama3.1")
gemma_difficulty   = load_and_prefix("CodeInsights_LLM_Difficulties/gemma-3-27b-it_item_difficulty.csv","gemma")
real_difficulty    = load_and_prefix("CodeInsights_LLM_Difficulties/real_item_difficulty.csv",  "real")
merged_difficulty = reduce(
    lambda left, right: pd.merge(left, right, on="item_id", how="outer"),
    [claude_difficulty, mistral_difficulty, gemini_difficulty, gpt_difficulty, llama_difficulty, gemma_difficulty, real_difficulty]
)
# 1) identify the comparison columns
difficulty_cols = [c for c in merged_difficulty.columns 
                if c.startswith("difficulty_") and c != "difficulty_real"]

# 2) compute correlations + p‑values one‑by‑one
results = []
for col in difficulty_cols:
    # drop rows where either real or this model is NaN
    sub = merged_difficulty[["difficulty_real", col]].dropna()

    # Pearson
    r, p_r = pearsonr(sub["difficulty_real"], sub[col])
    # Spearman
    s, p_s = spearmanr(sub["difficulty_real"], sub[col])

    results.append({
        "model":       col.replace("difficulty_", ""),
        "difficulty_corr":   r,
    })

# 3) assemble into a DataFrame
difficulty_corr_df = pd.DataFrame(results)
merged_df = ability_corr_df.merge(difficulty_corr_df)
data = [
    {'metric': 'Ability', 'model': 'gpt-4o','score': -0.123936},
    {'metric': 'Ability','model': 'gemini-2.5-pro',   'score': 0.273410},
    {'metric': 'Ability','model': 'claude-3-5','score': 0.359031},
    {'metric': 'Ability','model': 'mistral','score': 0.266511},
    {'metric': 'Ability','model': 'llama-3.1-8b','score': 0.029091},
    {'metric': 'Ability', 'model': 'gemma-3-27b','score': -0.106782},
    {'metric': 'Difficulty', 'model': 'gpt-4o','score': 0.152961},
    {'metric': 'Difficulty','model': 'gemini-2.5-pro',   'score': 0.288178},
    {'metric': 'Difficulty','model': 'claude-3-5','score': 0.233606},
    {'metric': 'Difficulty','model': 'mistral','score': 0.233606},
    {'metric': 'Difficulty','model': 'llama-3.1-8b','score': -0.203873},
    {'metric': 'Difficulty', 'model': 'gemma-3-27b','score': -0.287315},
]



