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

