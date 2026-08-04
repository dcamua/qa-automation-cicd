#=========================================================================
# IMPORT LIBRARIES
#=========================================================================
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

#=========================================================================
# LOAD DATA FROM EXCEL
#=========================================================================
df = pd.read_excel("mydatasets.xlsx")

#=========================================================================
# FEATURES & TARGET
#=========================================================================
x = df[
    [
        "Experience",
        "Completed_Projects",
        "Rating"
    ]
]
y = df["Grade"]
gender = df["Gender"]  # Sentitive attribute

#=========================================================================
# SPLIT DATA
#=========================================================================
x_train, x_test, y_train, y_test, gender_train, gender_test = train_test_split(
    x, y, gender,
    test_size=0.3,
    random_state=42,
    stratify=y
)

#=========================================================================
# TRAIN RANDOM FOREST CLASSIFIER
#=========================================================================
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

#=========================================================================
# PREDICTION
#=========================================================================
y_pred = model.predict(x_test)

#=========================================================================
# BIAS DETECTION REPORT (Before Mitigation)
#=========================================================================
results = pd.DataFrame({
    "Gender": gender_test.values,
    "TrueGrade": y_test.values,
    "PredictedGrade": y_pred
})

selection_rate = results.groupby("Gender")["PredictedGrade"].apply(lambda g: np.mean(g == "A"))
dp_diff = abs(selection_rate.max() - selection_rate.min())

print("\nBIAS DETECTION REPORT (Before Fix)")
print("=" * 60)

for gender, rate in selection_rate.items():
    print(f"{gender}: {rate: .2f}")
print(f"Demographic Parity Difference: {dp_diff: .2f}")

#=========================================================================
# FAIRNESS MITIGATION (Resampling)
#=========================================================================
male_df = df[df["Gender"] == "Male"]
female_df = df[df["Gender"] == "Female"]

if len(male_df) > len(female_df):
    female_df = resample(female_df, replace=True, n_samples=len(male_df), random_state=42)
else:
    male_df = resample(male_df, replace=True, n_samples=len(female_df), random_state=42)

balanced_df = pd.concat([male_df, female_df])

# Retrain model on balanced dataset
x_bal = balanced_df[["Experience", "Completed_Projects", "Rating"]]
y_bal = balanced_df["Grade"]
gender_bal = balanced_df["Gender"]

x_train_bal, x_test_bal, y_train_bal, y_test_bal, gender_train_bal, gender_test_bal = train_test_split(
    x_bal, y_bal, gender_bal,
    test_size=0.3,
    random_state=42,
    stratify=y_bal
)

model_bal = RandomForestClassifier(random_state=42)
model_bal.fit(x_train_bal, y_train_bal)
y_pred_bal = model_bal.predict(x_test_bal)

#=========================================================================
# BIAS DETECTION REPORT (After Resampling)
#=========================================================================
results_bal = pd.DataFrame({
    "Gender": gender_test_bal.values,
    "TrueGrade": y_test_bal.values,
    "PredictedGrade": y_pred_bal
})
selection_rate_bal = results_bal.groupby("Gender")["PredictedGrade"].apply(lambda g: np.mean(g == "A"))
dp_diff_bal = abs(selection_rate_bal.max() - selection_rate_bal.min())

print("\nBIAS DETECTION REPORT (After Fix)")
print("=" * 60)
for gender, rate in selection_rate_bal.items():
    print(f"{gender}: {rate: .2f}")
print(f"Demographic Parity Difference: {dp_diff_bal: .2f}")

#=========================================================================
# EXPLAINABILITY WITH SHAP
#=========================================================================
explainer = shap.TreeExplainer(model_bal)
shap_values = explainer.shap_values(x_test_bal)

print("\nMODEL DECISION EXPLANATION REPORT")
print("=" * 60)
print("Feature importance and contribution to predictions shown via SHAP plots.")

# Summary plot (global feature importance)
shap.summary_plot(shap_values, x_test_bal, plot_type="bar")

# Force plot for first prediction (local explanation)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], x_test_bal.iloc[0,:])