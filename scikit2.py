#=========================================================================
# Import Libraries
#=========================================================================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#=========================================================================
# CREATE DATASETS
#=========================================================================
data = {
    "Experience": [5, 2, 10, 6, 9, 10, 1, 3, 8, 7, 15, 2, 1, 5, 4, 8, 6, 5, 4, 11],
    "Completed_Projects": [10, 5, 7, 6, 21, 51, 9, 11, 2, 5, 10, 6, 1, 4, 10, 5, 6, 8, 12, 2],
    "Rating": [3, 1, 4, 5, 2, 2, 3, 5, 1, 2, 4, 5, 2, 2, 3, 4, 5, 2, 5, 3],
    "Grade": ["C", "A", "B", "A", "B", "C", "B", "C", "A", "A", "A", "C", "B", "B", "A", "C", "B", "C", "A", "B"]
}

df = pd.DataFrame(data)
#=========================================================================
# SPLIT DATA
#=========================================================================

x = df[["Experience", "Completed_Projects", "Rating"]]
y = df["Grade"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
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
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Accuracy: {accuracy: .2f}")

print(y_pred)