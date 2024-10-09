# %%
import pandas as pd


# %%
df = pd.read_csv("Car_Prices_Poland_Kaggle.csv")

# %%
df.dropna(how="any", axis=1, inplace=True)

# %%
df

# %%
df.drop("city", axis=1, inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder


df["province"] = LabelEncoder().fit_transform(df["province"])
df["mark"] = LabelEncoder().fit_transform(df["mark"])
df["model"] = LabelEncoder().fit_transform(df["model"])
df["fuel"] = LabelEncoder().fit_transform(df["fuel"])


# %%
x = df.drop("price", axis=1)
y = df["price"]

# %%
df.info()

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
rd = RandomForestRegressor()
rd.fit(X_train, y_train)

# %%
y_pred = rd.predict(X_test)

# %%
from sklearn.metrics import r2_score, mean_squared_error

# %%
print(r2_score(y_test, y_pred), f"{mean_squared_error(y_test, y_pred): 0.2f}")


