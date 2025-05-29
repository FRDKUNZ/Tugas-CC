import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Memuat Data
df = pd.read_csv('C:\\Users\\data_mining\\kelulusan.csv') 

# 2. Mengubah Label ('Lulus') ke dalam nilai numerik
le = preprocessing.LabelEncoder()
df['Lulus'] = le.fit_transform(df['Lulus'])  

# 3. Memilih Fitur dan Label
X = df[['Kehadiran', 'Nilai_Tugas', 'Nilai_UTS', 'Nilai_UAS']] 
y = df['Lulus']  # Label (variabel target)

# 4. Membagi Data menjadi Data Latih dan Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  


# 5. Melatih Model Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train) 

# 6. Memprediksi dan Mengevaluasi Model
y_pred = nb.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
print("Akurasi dari Naive Bayes Classifier: {:.2f}%".format(accuracy * 100)) 

# 7. Matriks Kebingungan (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Matriks Kebingungan')
plt.show()  

# 8. Probabilitas Prediksi
y_prob = nb.predict_proba(X_test)  
plt.figure(figsize=(8, 6))
plt.hist(y_prob[:, 1], bins=10, color='Blue', alpha=0.7)
plt.title('Distribusi Probabilitas untuk Kelas 1 (Lulus)')
plt.xlabel('Probabilitas')
plt.ylabel('Frekuensi')
plt.show()