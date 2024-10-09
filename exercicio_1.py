import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Contagem de tempo

t = time.time()

# Leitura do arquivo CSV
df = pd.read_csv("iris.csv")

# Preparação dos dados
x = df.drop("species", axis=1)
y = LabelEncoder().fit_transform(df["species"])

# Divisão dos dados
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)
smote = SMOTE(random_state=70)
x_treino_res, y_treino_res = smote.fit_resample(x_treino, y_treino)

# Treinamento do modelo
model = LogisticRegression()
model.fit(x_treino_res, y_treino_res)

# Previsões
y_pred = model.predict(x_teste)
y_pred_prob = model.predict_proba(x_teste)
# Avaliação
acuracia = accuracy_score(y_teste, y_pred)
matrix_confusao = confusion_matrix(y_teste, y_pred)

print(f"Acurácia: {acuracia}")
print(f"Matriz de Confusão: {matrix_confusao}")
print(f"Tempo gasto: {time.time()-t : 0.2f}s")
'''
Acurácia: 1.0 -> não consegui resolver this shit 😒
Matriz de Confusão: [[19  0  0]
                    [ 0  13  0]
                    [ 0  0  13]]
 Tempo gasto:  0.03s
'''