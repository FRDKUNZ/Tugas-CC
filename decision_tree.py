import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import preprocessing  
import matplotlib.pyplot as plt  
from sklearn import tree 

# 1. Memuat Data
df = pd.read_csv('C:/Users/data_mining/kelulusan.csv')  

# 2. Mengubah Label (Encoding kolom 'Lulus' ke nilai numerik)
le = preprocessing.LabelEncoder()  
df['Lulus'] = le.fit_transform(df['Lulus'])  

# 3. Memilih Fitur (X) dan Label (y)
X = df[['Kehadiran', 'Nilai_Tugas', 'Nilai_UTS', 'Nilai_UAS']]  
y = df['Lulus']  

# 4. Membagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  


# 5. Melatih Model Decision Tree
clf = DecisionTreeClassifier() 
clf.fit(X_train, y_train) 

# 6. Mengevaluasi Model (Mengukur Akurasi)
accuracy = clf.score(X_test, y_test)  
print("Accuracy of the Decision Tree Classifier: {:.2f}%".format(accuracy * 100))  


# 7. Visualisasi Decision Tree
plt.figure(figsize=(12, 8)) 
tree.plot_tree(
    clf,  
    feature_names=['Kehadiran', 'Nilai_Tugas', 'Nilai_UTS', 'Nilai_UAS'],
    filled=True, 
    fontsize=10 
)
plt.show()  
