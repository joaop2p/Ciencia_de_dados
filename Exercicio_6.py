from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
from matplotlib import pyplot as plt

# Carregar o DataFrame
df = pd.read_csv("./breast-cancer.csv", na_values="?")
df.dropna(how="any", inplace=True)

print(df)
x = df.drop("classe", axis=1)
y = LabelEncoder().fit_transform(df["classe"])

# Dividir os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.7, random_state=42)

# Aplicar SMOTE para balancear as classes
smote = SMOTE(random_state=70)
x_treino_res, y_treino_res = smote.fit_resample(x_treino, y_treino)

# Treinar o modelo Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(x_treino_res, y_treino_res)

# Fazer previsões e calcular a AUC-ROC
y_pred = gb.predict(x_teste)
y_pred_prob = gb.predict_proba(x_teste)[:, 1]
ar = roc_auc_score(y_teste, y_pred_prob)

print(f"AUC-ROC: {ar}")

# Parte feita com IA
fpr, tpr, thresholds = roc_curve(y_teste, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % ar)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
