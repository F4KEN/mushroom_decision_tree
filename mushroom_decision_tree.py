import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('dataset_24_mushroom.csv')


y = df['class'].map({'e'})
print(df.head())
print(df['class'].value_counts())
df_encoded = pd.get_dummies(df,drop_first = True)

X = df_encoded.drop("class_'p'",axis=1)
y = df_encoded["class_'p'"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("accuracy:", accuracy_score(y_test, y_pred))
print("classification report : \n", classification_report(y_test, y_pred))
print("confusion_matrix\n" , confusion_matrix(y_test,y_pred))

plt.figure(figsize=(20,20))
plot_tree(model, feature_names = X.columns, class_names= ["edible", "poisonous"], filled = True)
plt.title("mushroom classification tree")
plt.show()