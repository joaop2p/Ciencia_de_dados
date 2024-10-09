# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# %%
df = pd.read_csv("HousingData.csv")

# %%
df.dropna(how="any", axis=1, inplace=True)

# %%
df["TAX"].value_counts()

# %%
x  = df.drop("MEDV", axis=1)
y = df["MEDV"]

# %%
x_treino, x_test, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

# %%
model = LinearRegression()
model.fit(x_treino, y_treino)

# %%
y_pred = model.predict(x_test)

# %%
r2 = r2_score(y_teste, y_pred)
mae = mean_absolute_error(y_teste, y_pred)

# %%
print(r2,mae)


