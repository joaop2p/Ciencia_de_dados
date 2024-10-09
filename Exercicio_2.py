from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

df = pd.read_csv("./winequalityN.csv")

df.dropna(how="any",inplace=True) 

x = df.drop("type", axis = 1)
y = LabelEncoder().fit_transform(df["type"])

print(df["type"].value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)


smote = SMOTE(random_state=70)
x_treino_res, y_treino_res = smote.fit_resample(x_train, y_train)


kn = KNeighborsClassifier(5)
kn.fit(x_treino_res, y_treino_res)



y_pred = kn.predict(x_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1:.2f}')
