import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
# We use pandas to read the CSV file into a DataFrame.
try:
    df = pd.read_csv('Iris.csv')
except FileNotFoundError:
    print("Error: 'Iris.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Step 2: Prepare the data
# We separate the features (X) and the target variable (y).
# 'Species' is the target variable we want to predict.
# The 'Id' column is not a feature and should be dropped.
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

# Step 3: Split the data into training and testing sets
# We use a 80/20 split, meaning 80% of the data will be used for training
# and 20% for testing the model's performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("-" * 30)

# Step 4: Initialize and train the KNN model
# We create a KNeighborsClassifier instance with 'n_neighbors' (k) set to 5.
# This means the model will consider the 5 nearest neighbors to make a prediction.
knn = KNeighborsClassifier(n_neighbors=5)
print("Training the KNN model...")
knn.fit(X_train, y_train)

# Step 5: Make predictions on the test set
# We use the trained model to predict the species for the test data.
y_pred = knn.predict(X_test)

# Step 6: Evaluate the model's accuracy
# We compare the predicted species with the actual species to calculate accuracy.
accuracy = accuracy_score(y_test, y_pred)

print("Prediction and evaluation complete.")
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Example prediction (optional)
# You can use the trained model to predict the species for new, unseen data.
# Let's predict the species for a new flower with the following measurements:
# SepalLengthCm=6.0, SepalWidthCm=3.0, PetalLengthCm=4.0, PetalWidthCm=1.3
new_flower = pd.DataFrame([[6.0, 3.0, 4.0, 1.3]], columns=X.columns)
predicted_species = knn.predict(new_flower)
print("-" * 30)
print(f"New flower data: {new_flower.iloc[0].tolist()}")
print(f"Predicted species: {predicted_species[0]}")