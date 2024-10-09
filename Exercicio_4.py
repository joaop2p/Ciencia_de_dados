from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

# Carregar o DataFrame
df = pd.read_csv("./tested.csv")

# Verificar valores únicos em "Embarked" e depois remover colunas desnecessárias
print(df["Embarked"].value_counts())
df.drop(["Cabin", "Name", "Ticket", "Embarked"], axis=1, inplace=True)

# Remover linhas com valores nulos
df.dropna(how="any", axis=1, inplace=True)

# Codificar a coluna "Sex"
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Separar variáveis independentes e dependentes
x = df.drop("Survived", axis=1)
y = df["Survived"]

# Dividir os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.7, random_state=42)

# Aplicar SMOTE para balancear as classes
smote = SMOTE(random_state=70)
x_treino_res, y_treino_res = smote.fit_resample(x_treino, y_treino)

# Treinar o modelo de árvore de decisão
tree = DecisionTreeClassifier()
tree.fit(x_treino_res, y_treino_res)

# Fazer previsões e calcular a AUC-ROC
y_pred_prob = tree.predict_proba(x_teste)[:, 1]
ar = roc_auc_score(y_teste, y_pred_prob)

print(f"AUC-ROC: {ar}")
