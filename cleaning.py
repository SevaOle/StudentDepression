import pandas as pd

def get_clean(filename):
	df = pd.read_csv(filename)

	### NOTES
	"""
	After inspecting the data in Excel, I noticed:
	City: One "3" entry amongst cities
	Sleep duration, Dietary habits, Degree: "Others" category
	Dietary habits: "Others" category
	Financial stress: 3 "?" entries
	"""


	### CLEANING

	## DROPPING
	# I decided to drop:
	# - id: just an identifier, not useful
	# - city: might be slightly useful, but I don't have a good way to encode it
	# - profession: same as with city, I decided on numerical encoding
	# - degree: same, had to get rid of it
	df = df.drop(columns=["id", "City", "Profession", "Degree"])

	# Removing all "?" entries in financial stress
	df = df[df["Financial Stress"] != "?"]


	## CONVERTING TEXT TO NUMERICAL VALUES
	# Binary
	bin_cols = {
		"Gender": {"Male":0, "Female":1},
		"Have you ever had suicidal thoughts ?": {"No":0, "Yes":1},
		"Family History of Mental Illness": {"No":0, "Yes":1},
	}
	df.replace(bin_cols, inplace=True)

	# Categories
	sleep_map = {
		"'Less than 4 hours'": 3.5,
		"'4-5 hours'": 4.5,
		"'5-6 hours'": 5.5,
		"'6-7 hours'": 6.5,
		"'7-8 hours'": 7.5,
		"'More than 8 hours'": 9,
	}
	diet_map = {"Unhealthy":0, "Moderate":1, "Healthy":2}
	df["Sleep Duration"] = df["Sleep Duration"].map(sleep_map)
	df["Dietary Habits"] = df["Dietary Habits"].map(diet_map)

	df = df.dropna() # Drop rows with missing values

	#df.info()
	# Financial stress is still "object" type, so I need to convert it to numeric
	df["Financial Stress"] = pd.to_numeric(df["Financial Stress"], errors="coerce")

	df.info()
	return df