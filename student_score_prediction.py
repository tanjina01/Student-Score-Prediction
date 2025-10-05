import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("students.csv")
print("✅ Dataset Loaded Successfully!")
print(df.head())
print("\nDataset Info:")
print(df.describe())

# Step 2: Visualize Relationships
plt.figure(figsize=(10, 6))
plt.scatter(df["study_hours"], df["score"], color="blue", label="Study Hours vs Score")
plt.scatter(df["sleep_hours"], df["score"], color="green", label="Sleep Hours vs Score")
plt.scatter(df["participation"], df["score"], color="red", label="Participation vs Score")
plt.title("Student Performance Analysis")
plt.xlabel("Feature Values")
plt.ylabel("Final Score")
plt.legend()
plt.show()

# Step 3: Prepare Data for Training
X = df[["study_hours", "sleep_hours", "participation"]]
y = df["score"]

# Step 4: Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Visualize Actual vs Predicted Scores
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="purple")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Student Scores")
plt.grid(True)
plt.show()

# Step 9: Predict a New Student's Score
new_data = pd.DataFrame({
    "study_hours": [6],
    "sleep_hours": [7],
    "participation": [4]
})
predicted_score = model.predict(new_data)
print(f"\n🎯 Predicted score for a student with 6h study, 7h sleep, and participation 4: {predicted_score[0]:.2f}")
