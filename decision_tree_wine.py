#Import Librarise
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Datasets Load
a = pd.read_csv("D:\Document\Machine learning Intern\Task 1\WineQT.csv")
print("First 5 rows of the dataset:")
print(a.head())
print("\nDataset info:")
print(a.info())

#Preprocess the data
def quality_label(q):
    if q <= 5:
        return 0
    elif q <= 7:
        return 1
    else:
        return 2
a['quality_label'] = a['quality'].apply(quality_label)

#Features and target
X = a.drop(['quality', 'quality_label'], axis=1)
y = a['quality_label']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Decision Tree Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

#Visualization
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Low','Medium','High'],
    filled=True,
    rounded=True
)
plt.show()

#Model Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Low','Medium','High']))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
