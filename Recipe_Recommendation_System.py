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
