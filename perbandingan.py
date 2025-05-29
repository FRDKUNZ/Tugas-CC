
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Data
df = pd.read_csv('C:\\Users\\data_mining\\kelulusan.csv')

# 2. Convert Label (Encoding 'Lulus' to numeric values)
le = preprocessing.LabelEncoder()
df['Lulus'] = le.fit_transform(df['Lulus'])  # 'Ya' = 1, 'Tidak' = 0

# 3. Select Features and Target
X = df[['Kehadiran', 'Nilai_Tugas', 'Nilai_UTS', 'Nilai_UAS']]
y = df['Lulus']

# 4. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 6. Predict
y_pred = clf.predict(X_test)

# 7. Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 8. Output the Results
print("Evaluation Metrics for Decision Tree Classifier:")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")
