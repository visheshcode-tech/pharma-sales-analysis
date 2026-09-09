"""
Drug Effectiveness Analysis
Author: Vishesh Pandey
Description: Analyzes patient response data across multiple drugs to evaluate
recovery rates, side-effect frequency, and efficacy scores, then ranks drugs
using a composite data-driven score.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. SIMULATE A RAW PATIENT-LEVEL DATASET (500+ records, with real-world mess)
# ----------------------------------------------------------------------
n = 560
drugs = ["Drug A", "Drug B", "Drug C", "Drug D", "Drug E"]
disease_groups = ["Hypertension", "Type-2 Diabetes", "Asthma", "Arthritis"]

# Give each drug a "true" underlying effectiveness/side-effect profile
drug_profile = {
    "Drug A": {"recovery_p": 0.78, "side_effect_p": 0.12, "efficacy_mu": 8.1},
    "Drug B": {"recovery_p": 0.65, "side_effect_p": 0.22, "efficacy_mu": 6.9},
    "Drug C": {"recovery_p": 0.83, "side_effect_p": 0.09, "efficacy_mu": 8.6},
    "Drug D": {"recovery_p": 0.58, "side_effect_p": 0.31, "efficacy_mu": 6.1},
    "Drug E": {"recovery_p": 0.71, "side_effect_p": 0.17, "efficacy_mu": 7.4},
}

rows = []
for i in range(n):
    drug = np.random.choice(drugs)
    profile = drug_profile[drug]
    age = int(np.clip(np.random.normal(48, 15), 18, 90))
    recovered = np.random.rand() < profile["recovery_p"]
    side_effect = np.random.rand() < profile["side_effect_p"]
    efficacy = np.clip(np.random.normal(profile["efficacy_mu"], 1.2), 0, 10)
    treatment_days = int(np.clip(np.random.normal(21, 6), 5, 60))

    rows.append({
        "patient_id": f"P{1000+i}",
        "drug": drug,
        "disease_group": np.random.choice(disease_groups),
        "age": age,
        "gender": np.random.choice(["Male", "Female", "male", "FEMALE"]),  # messy on purpose
        "treatment_days": treatment_days,
        "recovered": "Yes" if recovered else "No",
        "side_effect_reported": "Y" if side_effect else "N",
        "efficacy_score": round(efficacy, 1),
    })

df_raw = pd.DataFrame(rows)

# Inject some realistic messiness: missing values & duplicate rows
missing_idx = np.random.choice(df_raw.index, size=25, replace=False)
df_raw.loc[missing_idx, "efficacy_score"] = np.nan
df_raw = pd.concat([df_raw, df_raw.sample(8, random_state=1)], ignore_index=True)

df_raw.to_csv("/home/claude/project1/patient_data_raw.csv", index=False)
print(f"Raw dataset shape: {df_raw.shape}")

# ----------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------
df = df_raw.drop_duplicates(subset="patient_id").copy()
df["gender"] = df["gender"].str.strip().str.capitalize()
df["gender"] = df["gender"].replace({"Female": "Female", "Male": "Male"})
df["recovered_flag"] = df["recovered"].map({"Yes": 1, "No": 0})
df["side_effect_flag"] = df["side_effect_reported"].map({"Y": 1, "N": 0})

# Fill missing efficacy scores with the drug-wise median (more accurate than global mean)
df["efficacy_score"] = df.groupby("drug")["efficacy_score"].transform(
    lambda x: x.fillna(x.median())
)

df.to_csv("/home/claude/project1/patient_data_clean.csv", index=False)
print(f"Cleaned dataset shape: {df.shape}")

# ----------------------------------------------------------------------
# 3. GROUP-WISE ANALYSIS
# ----------------------------------------------------------------------
summary = df.groupby("drug").agg(
    patients=("patient_id", "count"),
    recovery_rate=("recovered_flag", "mean"),
    side_effect_rate=("side_effect_flag", "mean"),
    avg_efficacy=("efficacy_score", "mean"),
    avg_treatment_days=("treatment_days", "mean"),
).round(3)

summary["recovery_rate_pct"] = (summary["recovery_rate"] * 100).round(1)
summary["side_effect_rate_pct"] = (summary["side_effect_rate"] * 100).round(1)

# ----------------------------------------------------------------------
# 4. COMPOSITE RANKING SCORE
#    score = recovery_rate*0.45 + (efficacy/10)*0.4 - side_effect_rate*0.15
# ----------------------------------------------------------------------
summary["composite_score"] = (
    summary["recovery_rate"] * 0.45
    + (summary["avg_efficacy"] / 10) * 0.4
    - summary["side_effect_rate"] * 0.15
).round(3)

summary = summary.sort_values("composite_score", ascending=False)
summary["rank"] = range(1, len(summary) + 1)

print("\n=== DRUG PERFORMANCE SUMMARY ===")
print(summary[["patients", "recovery_rate_pct", "side_effect_rate_pct",
                "avg_efficacy", "composite_score", "rank"]])

summary.to_csv("/home/claude/project1/drug_summary.csv")

# ----------------------------------------------------------------------
# 5. VISUALIZATIONS
# ----------------------------------------------------------------------
order = summary.index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Recovery rate
sns.barplot(x=summary.index, y=summary["recovery_rate_pct"], order=order,
            palette="Blues_d", ax=axes[0, 0])
axes[0, 0].set_title("Recovery Rate by Drug (%)")
axes[0, 0].set_ylabel("Recovery Rate (%)")
axes[0, 0].set_xlabel("")

# Side-effect rate
sns.barplot(x=summary.index, y=summary["side_effect_rate_pct"], order=order,
            palette="Reds_d", ax=axes[0, 1])
axes[0, 1].set_title("Side-Effect Frequency by Drug (%)")
axes[0, 1].set_ylabel("Side-Effect Rate (%)")
axes[0, 1].set_xlabel("")

# Efficacy score distribution
sns.boxplot(x="drug", y="efficacy_score", data=df, order=order,
            palette="Greens_d", ax=axes[1, 0])
axes[1, 0].set_title("Efficacy Score Distribution by Drug")
axes[1, 0].set_ylabel("Efficacy Score (0-10)")
axes[1, 0].set_xlabel("")

# Composite ranking
sns.barplot(x=summary.index, y=summary["composite_score"], order=order,
            palette="Purples_d", ax=axes[1, 1])
axes[1, 1].set_title("Overall Composite Ranking Score")
axes[1, 1].set_ylabel("Composite Score")
axes[1, 1].set_xlabel("")

plt.tight_layout()
plt.savefig("/home/claude/project1/drug_analysis_dashboard.png", dpi=150)
print("\nSaved dashboard chart.")

# Disease-group breakdown chart
fig2, ax2 = plt.subplots(figsize=(10, 5))
pivot = df.pivot_table(index="disease_group", columns="drug",
                        values="recovered_flag", aggfunc="mean") * 100
pivot.plot(kind="bar", ax=ax2, colormap="tab10")
ax2.set_title("Recovery Rate (%) by Disease Group and Drug")
ax2.set_ylabel("Recovery Rate (%)")
ax2.set_xlabel("Disease Group")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("/home/claude/project1/disease_group_breakdown.png", dpi=150)
print("Saved disease-group breakdown chart.")

# ----------------------------------------------------------------------
# 6. TOP RECOMMENDATION
# ----------------------------------------------------------------------
top_drug = summary.index[0]
print(f"\nTOP RECOMMENDED DRUG: {top_drug}")
print(f"  Recovery Rate: {summary.loc[top_drug, 'recovery_rate_pct']}%")
print(f"  Side-Effect Rate: {summary.loc[top_drug, 'side_effect_rate_pct']}%")
print(f"  Avg Efficacy: {summary.loc[top_drug, 'avg_efficacy']}/10")
