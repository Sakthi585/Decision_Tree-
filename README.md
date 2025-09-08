Decision Tree Classifier
This project demonstrates the implementation of a Decision Tree Classifier using scikit-learn. The model is applied to the Wine dataset, which contains chemical analysis of wines grown in the same region in Italy 
but derived from three different cultivars.

The notebook includes:
Data preprocessing
Model training
Visualization of the decision tree
Performance evaluation

Dataset
Source: Built-in load_wine dataset from scikit-learn
Samples: 178
Features: 13 chemical attributes (e.g., alcohol, flavanoids, color intensity)
Target Classes: 3 types of wine
Steps Performed
Import necessary libraries (pandas, matplotlib, sklearn).
Load the wine dataset.
Split the data into training and testing sets.
Train a DecisionTreeClassifier.
Visualize the decision tree using plot_tree.
Evaluate performance using:
Accuracy Score
Confusion Matrix
Classification Report

Results
The decision tree model successfully classifies wines into 3 categories.
Performance metrics (accuracy, precision, recall, f1-score) are displayed in the notebook.
A visualization of the trained decision tree helps understand decision paths.

Technologies Used
Python
Scikit-learn
Matplotlib
Pandas
