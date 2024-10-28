# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the Tips dataset from seaborn's library
df = sns.load_dataset("tips")

# Display first few rows and summary statistics
print(df.head()) #Prints first few rows of the dataset
print(df.describe()) #Print Summary Statistics 

# 1. Histogram of Total Bill
plt.figure(figsize=(8, 6)) #Initializes a new figure with a specified size of 8x6 inches.
sns.histplot(df['total_bill'], kde=True, color='purple') #
plt.title("Distribution of Total Bill") #Adds a title to the plot.
plt.xlabel("Total Bill ($)") #Labels the x-axis as "Total Bill ($)"
plt.ylabel("Frequency") #Labels the y-axis as "Frequency".
plt.show() #Displays the plot 
plt.clf()  # Clear figure

# 2. Boxplot of Tips by Day with hue as Sex
plt.figure(figsize=(8, 6))
sns.boxplot(x='day', y='tip', hue='sex', data=df, palette="Set3", dodge=True)
plt.title("Tip Amount by Day and Gender") 
plt.xlabel("Day of the Week") #Labels the x-axis as "Day of the Week".
plt.ylabel("Tip ($)") #Labels the y-axis as "Tip ($)".
plt.legend(title="Gender")
plt.show()
plt.clf()  # Clear figure

# 3. Scatterplot of Total Bill vs. Tip Amount, colored by Sex
plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_bill', y='tip', hue='sex', style='sex', data=df)
plt.title("Total Bill vs. Tip Amount (Colored by Gender)")
plt.xlabel("Total Bill ($)") #Labels the x-axis as "Total Bill ($)"
plt.ylabel("Tip ($)") #Labels the y-axis as "Tip ($)"
plt.legend(title="Gender") #Adds a legend titled "Gender" to identify colors and shapes by gender.
plt.show()
plt.clf()  # Clear figure

# 4. Heatmap of Correlation between Numerical Features
# Selecting only numerical columns for correlation matrix
numerical_df = df.select_dtypes(include='number')
plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Tips Dataset")
plt.show()
plt.clf()  # Clear figure

# 5. Line Plot of Average Total Bill by Party Size
avg_bill_by_size = df.groupby("size")["total_bill"].mean()
plt.figure(figsize=(8, 6))
sns.lineplot(x=avg_bill_by_size.index, y=avg_bill_by_size.values, marker='o', color='green')
plt.title("Average Total Bill by Party Size")
plt.xlabel("Party Size")
plt.ylabel("Average Total Bill ($)")
plt.show()
plt.clf()  # Clear figure



