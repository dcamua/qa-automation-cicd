#=========================================================================
# Import Libraries
#=========================================================================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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

#=========================================================================
# SPLIT DATA
#=========================================================================
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

#=========================================================================
# TRAIN RANDOM FOREST CLASSIFIER
#=========================================================================
model = RandomForestClassifier()
model.fit(x_train, y_train)

#=========================================================================
# PREDICTION
#=========================================================================
y_pred = model.predict(x_test)

#=========================================================================
# GENERATE REPORT
#=========================================================================
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

accuracy = accuracy_score(y_test, y_pred)

#=========================================================================
# DISPLAY RESULT
#=========================================================================
print("\nTEST REPORT ON SCIKIT-LEARN CLASSIFIER")
print("=" * 60)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy: .2f}")
