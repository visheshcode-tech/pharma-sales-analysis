"""
Hospital Patient Data Analysis
Author: Vishesh Pandey
Description: Analyzes a multi-department hospital dataset to study treatment
cost patterns and recovery trends, identifying high-cost/low-recovery
scenarios and opportunities for resource allocation improvement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
np.random.seed(7)

# ----------------------------------------------------------------------
# 1. SIMULATE A RAW HOSPITAL DATASET (multi-department, real-world mess)
# ----------------------------------------------------------------------
n = 620
departments = ["Cardiology", "Orthopedics", "Oncology", "General Medicine", "Neurology"]

dept_profile = {
    "Cardiology": {"cost_mu": 185000, "cost_sd": 45000, "recovery_p": 0.80, "los_mu": 6},
    "Orthopedics": {"cost_mu": 120000, "cost_sd": 30000, "recovery_p": 0.86, "los_mu": 5},
    "Oncology": {"cost_mu": 310000, "cost_sd": 90000, "recovery_p": 0.58, "los_mu": 12},
    "General Medicine": {"cost_mu": 65000, "cost_sd": 20000, "recovery_p": 0.88, "los_mu": 4},
    "Neurology": {"cost_mu": 210000, "cost_sd": 55000, "recovery_p": 0.66, "los_mu": 9},
}

rows = []
for i in range(n):
    dept = np.random.choice(departments)
    profile = dept_profile[dept]
    age = int(np.clip(np.random.normal(52, 18), 1, 95))
    cost = max(5000, np.random.normal(profile["cost_mu"], profile["cost_sd"]))
    los = max(1, int(np.random.normal(profile["los_mu"], 2.5)))
    recovered = np.random.rand() < profile["recovery_p"]

    rows.append({
        "patient_id": f"H{2000+i}",
        "department": dept,
        "age": age,
        "gender": np.random.choice(["M", "F", "m", "Female "]),  # messy on purpose
        "length_of_stay_days": los,
        "treatment_cost": round(cost, 0),
        "recovery_status": "Recovered" if recovered else "Not Recovered",
        "readmitted_30d": np.random.choice(["Yes", "No"], p=[0.12, 0.88]),
    })

df_raw = pd.DataFrame(rows)

# Inject missing values & duplicates
missing_idx = np.random.choice(df_raw.index, size=20, replace=False)
df_raw.loc[missing_idx, "treatment_cost"] = np.nan
df_raw = pd.concat([df_raw, df_raw.sample(10, random_state=2)], ignore_index=True)

df_raw.to_csv("/home/claude/project2/hospital_data_raw.csv", index=False)
print(f"Raw dataset shape: {df_raw.shape}")

# ----------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------
df = df_raw.drop_duplicates(subset="patient_id").copy()
df["gender"] = df["gender"].str.strip().str.upper().str[0]
df["gender"] = df["gender"].map({"M": "Male", "F": "Female"})
df["recovered_flag"] = (df["recovery_status"] == "Recovered").astype(int)
df["readmitted_flag"] = (df["readmitted_30d"] == "Yes").astype(int)

# Fill missing cost with department-wise median
df["treatment_cost"] = df.groupby("department")["treatment_cost"].transform(
    lambda x: x.fillna(x.median())
)

# Feature engineering: cost per day, age bracket
df["cost_per_day"] = (df["treatment_cost"] / df["length_of_stay_days"]).round(0)
df["age_bracket"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100],
                            labels=["0-18", "19-35", "36-60", "60+"])

df.to_csv("/home/claude/project2/hospital_data_clean.csv", index=False)
print(f"Cleaned dataset shape: {df.shape}")

# ----------------------------------------------------------------------
# 3. DEPARTMENT-WISE ANALYSIS
# ----------------------------------------------------------------------
summary = df.groupby("department").agg(
    patients=("patient_id", "count"),
    avg_cost=("treatment_cost", "mean"),
    avg_los=("length_of_stay_days", "mean"),
    recovery_rate=("recovered_flag", "mean"),
    readmission_rate=("readmitted_flag", "mean"),
).round(2)

summary["avg_cost"] = summary["avg_cost"].round(0)
summary["recovery_rate_pct"] = (summary["recovery_rate"] * 100).round(1)
summary["readmission_rate_pct"] = (summary["readmission_rate"] * 100).round(1)

# Efficiency flag: high cost + low recovery = needs attention
cost_median = summary["avg_cost"].median()
recovery_median = summary["recovery_rate"].median()
summary["efficiency_flag"] = np.where(
    (summary["avg_cost"] > cost_median) & (summary["recovery_rate"] < recovery_median),
    "High-Cost / Low-Recovery (Review Needed)",
    "Acceptable"
)

summary = summary.sort_values("avg_cost", ascending=False)
print("\n=== DEPARTMENT PERFORMANCE SUMMARY ===")
print(summary[["patients", "avg_cost", "avg_los", "recovery_rate_pct",
                "readmission_rate_pct", "efficiency_flag"]])

summary.to_csv("/home/claude/project2/hospital_summary.csv")

# ----------------------------------------------------------------------
# 4. VISUALIZATIONS
# ----------------------------------------------------------------------
order = summary.index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.barplot(x=summary.index, y=summary["avg_cost"], order=order,
            hue=summary.index, palette="Oranges_d", legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Average Treatment Cost by Department (Rs.)")
axes[0, 0].set_xlabel("")
axes[0, 0].tick_params(axis='x', rotation=20)

sns.barplot(x=summary.index, y=summary["recovery_rate_pct"], order=order,
            hue=summary.index, palette="Greens_d", legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Recovery Rate by Department (%)")
axes[0, 1].set_xlabel("")
axes[0, 1].tick_params(axis='x', rotation=20)

sns.boxplot(x="department", y="length_of_stay_days", data=df, order=order,
            hue="department", palette="Blues_d", legend=False, ax=axes[1, 0])
axes[1, 0].set_title("Length of Stay Distribution by Department")
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis='x', rotation=20)

sns.barplot(x=summary.index, y=summary["readmission_rate_pct"], order=order,
            hue=summary.index, palette="Reds_d", legend=False, ax=axes[1, 1])
axes[1, 1].set_title("30-Day Readmission Rate by Department (%)")
axes[1, 1].set_xlabel("")
axes[1, 1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig("/home/claude/project2/hospital_analysis_dashboard.png", dpi=150)
print("\nSaved dashboard chart.")

# Cost vs recovery scatter (bubble = patient volume)
fig2, ax2 = plt.subplots(figsize=(9, 6))
sizes = summary["patients"] * 3
scatter = ax2.scatter(summary["recovery_rate_pct"], summary["avg_cost"],
                       s=sizes, alpha=0.6, c=range(len(summary)), cmap="viridis")
for dept in summary.index:
    ax2.annotate(dept, (summary.loc[dept, "recovery_rate_pct"], summary.loc[dept, "avg_cost"]),
                 fontsize=9, ha="center", va="bottom")
ax2.axvline(recovery_median * 100, color="gray", linestyle="--", alpha=0.5)
ax2.axhline(cost_median, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("Recovery Rate (%)")
ax2.set_ylabel("Average Treatment Cost (Rs.)")
ax2.set_title("Cost vs. Recovery Rate by Department\n(bubble size = patient volume)")
plt.tight_layout()
plt.savefig("/home/claude/project2/cost_vs_recovery.png", dpi=150)
print("Saved cost-vs-recovery chart.")

# ----------------------------------------------------------------------
# 5. KEY FINDING
# ----------------------------------------------------------------------
flagged = summary[summary["efficiency_flag"] != "Acceptable"]
print("\nDEPARTMENTS FLAGGED FOR REVIEW (high cost, low recovery):")
print(flagged.index.tolist())
