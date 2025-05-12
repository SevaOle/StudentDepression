from cleaning import get_clean
from report import report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

import joblib

# === data prep ===
print("CLEANING DATA")
df = get_clean("student_depression_dataset.csv")
print("\n\n\n")

X = df.drop("Depression", axis=1)
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=777
)


### 1) LOGISTIC REGRESSION
regression_model = make_pipeline(
    StandardScaler(), # centers data so that the mean is 0
    LogisticRegression(max_iter=1000)
)
regression_model.fit(X_train, y_train)

print("=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, regression_model.predict(X_test)))



### 2) DECISION TREE

decisionTree_model = DecisionTreeClassifier(random_state=777)
decisionTree_model.fit(X_train, y_train)

print("\n=== DECISION TREE ===")
print(classification_report(y_test, decisionTree_model.predict(X_test)))



# 3) RANDOM FOREST
randomForest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=777
)
randomForest_model.fit(X_train, y_train)

print("\n=== RANDOM FOREST ===")
print(classification_report(y_test, randomForest_model.predict(X_test)))


### FINAL
## Report
# Logistic regression showed the best f1 score, so I will use it.
report(regression_model, X_train, y_train, X_test, y_test)

## Save the model using joblib
joblib.dump(regression_model, "logistic_regression_model.pkl")