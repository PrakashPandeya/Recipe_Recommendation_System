"""
Original file is located at
    https://colab.research.google.com/drive/1K0SY5r6Oq_x2ay4Iu1uZGqcV3Bjmks22

Dataset Name = Receipe Reviews and User Feedback
Dataset Link = https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset
"""

#importing Necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#importing the dataset
df = pd.read_csv("Recipe Reviews and User Feedback Dataset.csv")

#Reading the dataset
#Displaying Head of our dataset
display(df.head())

#Displaying tail of our dataset
display(df.tail())

#Displaying sample of our dataset
print(df.sample())

#Showing the total rows and columns:
print(df.shape)
#(rows,columns)

#Checking data types of each columns:
print(df.dtypes)

#Preprocessign Of our data
# Check missing values
print(df.isnull().sum())


"""# Performing KNN

To perform KNN , among the 15 columns:
We can use columns: recipe_number, recipe_code, user_reputation, reply_count, thumbs_up, thumbs_down, stars, best_score as Features . And for the Target columns we  can use column receipe_name.  Also the columns like unnamed, comment_id, user_id, user_name, text , created_at are not necessary  so I determined to remove these unnecessary columns.
"""

#Removing Unnecessary columns
unnecessary_columns = ["Unnamed: 0", "comment_id","recipe_code","user_id", "user_name", "text", "created_at"]
training_data = df.drop(columns=unnecessary_columns, axis=1) #axis=1 as we are dropping the columns

# Columns after dropping unnecessary_columns
print(training_data)


# Extract column names
columns = training_data.columns

# Assign equal weights to each column
values = [1] * len(columns)

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=columns, autopct='%1.1f%%', startangle=140)
plt.title('Columns of Cleaned DataFrame')
plt.show()

"""The percentage (12.5%) appears because the pie chart is dividing the total "weight" equally among all columns in the DataFrame. Since there are 8 columns in our cleaned DataFrame, each slice gets an equal share."""

# Selecting Key Features for Recommendation
features = ["recipe_number","user_reputation", "reply_count", "thumbs_up", "thumbs_down", "stars", "best_score"]
X = training_data[features]
print("Selected features:")
display(X.head())
print("\nBasic statistics of features:")
display(X.describe())

#Visualizing the Features
training_data[["recipe_number","user_reputation", "reply_count", "thumbs_up", "thumbs_down", "stars", "best_score"]].hist(bins=10, figsize=(10, 10))
plt.suptitle("Feature Distributions")
plt.show()

#Selecting Target Column for Recommendation
y = training_data["recipe_name"]

# Displaying the shape(i.e. number of rows and columns) of our cleaned dataset
print("\nFinal shape of features (X):", X.shape)
print("Final shape of target (y):", y.shape)

#Splitting the dataset into train_test_split
from sklearn.model_selection import train_test_split
# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

#Checking the ranges of our features column
print(X.describe())






"""Visualization Correlation Heatmap
Purpose: Identify relationships between numeric features.
"""

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

"""Here, after looking at the ranges (min and max values) of our features columns, we can see the difference in maximum value is different in different columns . So we have to do sclaing here as Scaling will ensure that all features contribute equally to the distance calculations in KNN."""

#Here Scaling the data of my features columns as
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on test set
y_pred = knn.predict(X_test)

# Evaluate the model
Knn_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {Knn_accuracy:.2f}")







# Function to recommend recipes
def recommend_recipes(user_input, n_recommendations=5):
    # Scale the input using the same scaler
    scaled_input = scaler.transform([user_input])

    # Find the k nearest neighbors
    distances, indices = knn.kneighbors(scaled_input, n_neighbors=n_recommendations)

    # Get the recommended recipe names
    recommendations = y[indices[0]]

    return recommendations

# Example usage:
# sample_input = [recipe_number, user_reputation, reply_count, thumbs_up, thumbs_down, stars, best_score]
sample_input = [1, 600, 2, 10, 1, 4, 8]
recommendations = recommend_recipes(sample_input)
print("\nRecommended Recipes:")
for i, recipe in enumerate(recommendations, 1):
    print(f"{i}. {recipe}")



"""Using Next Algortihm - Random Forest using the same Features and Target.

We used Random Forest because Random Forest can enhance predictions, especially for multi-class classification problems (e.g., predicting a specific recipe name).
"""

#importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initializing the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)





# Evaluate performance
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rf))  # For Random Forest



models = ['KNN', 'Random Forest']
accuracies = [Knn_accuracy, accuracy_score(y_test, y_pred_rf)]
plt.bar(models, accuracies, color=['blue', 'green'], edgecolor='black')
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)
plt.show()

# One advantage of Random Forest is that it can show feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))


feature_importance = feature_importance.sort_values('importance', ascending=True)
plt.figure(figsize=(8, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='lightcoral', edgecolor='black')
plt.title("Feature Importance (Random Forest)", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.show()



# Function to recommend recipes using Random Forest
def recommend_recipes_rf(user_input, n_recommendations=5):
    # Scale the input using the same scaler
    scaled_input = scaler.transform([user_input])

    # Get probability predictions for all classes
    probabilities = rf.predict_proba(scaled_input)

    # Get top N recipe indices
    top_n_indices = np.argsort(probabilities[0])[-n_recommendations:][::-1]

    # Get the recommended recipe names
    recommendations = [rf.classes_[idx] for idx in top_n_indices]

    return recommendations

# Example usage:
# sample_input = [recipe_number, user_reputation, reply_count, thumbs_up, thumbs_down, stars, best_score]
sample_input = [1, 100, 2, 10, 1, 4, 800]
recommendations = recommend_recipes_rf(sample_input)
print("\nRecommended Recipes:")
for i, recipe in enumerate(recommendations, 1):
    print(f"{i}. {recipe}")


"""Now, Using the Next Algorithm, K_means Clustering. it is used to group similar data points. It helps cluster recipes or users based on their features (e.g., thumbs_up, stars, etc.), enabling recommendations within a cluster."""

#importing necessary Libraries
from sklearn.cluster import KMeans

# Initializing KMeans with a predefined number of clusters
kmeans = KMeans(n_clusters=5, random_state=42)

# Fitting the model to the data
kmeans.fit(X)

# Assigning clusters to each recipe
clusters = kmeans.labels_
training_data['Cluster'] = clusters

# Displaying the random with cluster information
print(training_data.sample(10))

plt.figure(figsize=(10, 8))
for cluster_id in training_data['Cluster'].unique():
    cluster_data = training_data[training_data['Cluster'] == cluster_id]
    plt.scatter(cluster_data['thumbs_up'], cluster_data['stars'], label=f'Cluster {cluster_id}', s=50)
plt.title("K-Means Clustering of Recipes", fontsize=14)
plt.xlabel("Thumbs Up", fontsize=12)
plt.ylabel("Stars", fontsize=12)
plt.legend()
plt.show()

distortions = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), distortions, marker='o', linestyle='--', color='purple')
plt.title("Elbow Method for Optimal Clusters", fontsize=14)
plt.xlabel("Number of Clusters", fontsize=12)
plt.ylabel("Distortion", fontsize=12)
plt.show()


# Example Recommendation:
# Find all recipes in the same cluster as a user's liked recipe
liked_recipe = training_data[training_data['recipe_name'] == 'Seafood Lasagna']
cluster_id = liked_recipe['Cluster'].values[0]
recommendations = training_data[training_data['Cluster'] == cluster_id]
print("Recommended Recipes in the same cluster:\n", recommendations[['recipe_name']])

from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), silhouette_scores, marker='o', linestyle='--', color='purple')
plt.title("Silhouette Scores for KMeans Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()



sample_input = [1, 600, 2, 10, 1, 4, 8]
recommendations = recommend_recipes(sample_input)
print("\nRecommended Recipes:")
for i, recipe in enumerate(recommendations, 1):
    print(f"{i}. {recipe}")
