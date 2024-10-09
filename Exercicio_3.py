
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, precision_recall_curve
    )
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from imblearn.over_sampling import SMOTE
import random

df = read_csv("./breast-cancer.csv", na_values="?")
df.dropna(how="any",inplace=True)

x = df.drop("classe", axis=1).astype(int)
y = LabelEncoder().fit_transform(df["classe"])

x_treino,x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.3, random_state=random.randint(40,50))

smote = SMOTE(random_state=70)
x_treino_res, y_treino_res = smote.fit_resample(x_treino, y_treino)

svm = SVC(kernel='linear', probability=True) #-> TODO Ignorar o sonarlimit
svm.fit(x_treino_res, y_treino_res)

y_pred = svm.predict(x_teste)
y_prob = svm.predict_proba(x_teste)[:, 1]

precision = precision_score(y_teste, y_pred, average='weighted')
recall = recall_score(y_teste, y_pred, average='weighted')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')


