# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

# %%
df = pd.read_csv("./iris.csv")

# %%
df["species"]  = LabelEncoder().fit_transform(df["species"])

# %%
df

# %%
x = df.drop("species", axis=1)
y = df["species"]

# %%
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.3, random_state=42)

# %%
smote = SMOTE(random_state=70)
x_treino_res, y_treino_res = smote.fit_resample(x_treino, y_treino)

# %%
nv = GaussianNB()
nv.fit(x_treino_res,y_treino_res)

# %%
from sklearn.metrics import confusion_matrix

y_pred_treino_prob = nv.predict_proba(x_treino)
y_pred_treino = nv.predict(x_treino)


print(average_precision_score(y_treino, y_pred_treino_prob, average="macro"))
print(confusion_matrix(y_treino,y_pred_treino ))

# %%



