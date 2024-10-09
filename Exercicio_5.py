# %%
import pandas as pd
from sklearn.datasets import load_digits


# %%
digits = load_digits()

# %%
df = pd.DataFrame(data=digits.data, columns=digits.feature_names)

# %%
df.dropna(how="any", axis=1, inplace=True)

# %%
df['target'] = digits.target

# %%
df

# %%
x = df.drop("target", axis=1)
y = df["target"]

# %%
from sklearn.model_selection import train_test_split

# %%
x_treino, x_test, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf = RandomForestClassifier()
rf.fit(x_treino, y_treino)

# %%
y_pred = rf.predict(x_test)

# %%
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# %%
print(f1_score(y_teste, y_pred, average="macro"))
print(confusion_matrix(y_teste, y_pred))


