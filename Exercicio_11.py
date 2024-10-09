# %%
import pandas as pd

# %%
df = pd.read_csv("HousingData.csv")

# %%
df.dropna(how="any", axis=1, inplace=True)

# %%
df

# %%
x = df.drop("MEDV", axis=1)
y = df["MEDV"]

# %%
df.info()

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
from sklearn.svm import SVR

# %%
kn = SVR()
kn.fit(X_train, y_train)

# %%
y_pred = kn.predict(X_test)

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %%
print(mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred))


