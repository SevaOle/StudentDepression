import easygui as eg
import pandas as pd
import joblib

### GUI

## load model
model = joblib.load("logistic_regression_model.pkl")
cols = list(model.feature_names_in_)

## Getting user input
gender = eg.choicebox("Gender?", choices=["Male", "Female"])
age = eg.enterbox("Age:")
acad = eg.enterbox("Academic pressure (1‑5):")
work_press = eg.enterbox("Work pressure (1‑5):")
gpa = eg.enterbox("GPA (0‑4):")
cgpa = float(gpa) * 2.5 # convert gpa to cgpa used in the dataset
study_sat = eg.enterbox("Study satisfaction 1‑5:")
job_sat = eg.enterbox("Job satisfaction (1‑5):")
sleep = eg.enterbox("Sleep hours (example: 5.5):")
diet = eg.choicebox("Dietary habits?", choices=["Unhealthy", "Moderate", "Healthy"])
suicidal = eg.choicebox("Suicidal thoughts?", choices=["Yes", "No"])
work_hrs = eg.enterbox("Work/study hours per day:")
fin_stress = eg.enterbox("Financial stress (1‑5):")
fam_hist = eg.choicebox("Family mental‑illness history?", choices=["Yes", "No"])

## convert user data to numerical values
data = {
    "Gender": 0 if gender == "Male" else 1,
    "Age": float(age),
    "Academic Pressure": float(acad),
    "Work Pressure": float(work_press),
    "CGPA": cgpa,
    "Study Satisfaction": float(study_sat),
    "Job Satisfaction": float(job_sat),
    "Sleep Duration": float(sleep),
    "Dietary Habits": {"Unhealthy":0, "Moderate":1, "Healthy":2}[diet],
    "Have you ever had suicidal thoughts ?": 1 if suicidal == "Yes" else 0,
    "Work/Study Hours": float(work_hrs),
    "Financial Stress": float(fin_stress),
    "Family History of Mental Illness": 1 if fam_hist == "Yes" else 0,
}

## predict, display result
row = pd.DataFrame([data]).reindex(cols, axis=1)
pred = model.predict(row)[0]
eg.msgbox("Likely depressed" if pred else "More likely to not be depressed")
