
# %%
import pandas as pd

# %%
df = pd.read_csv("./HousingData.csv")

# %%
df.dropna(how="any", axis=1, inplace=True)

# %%
df

# %%
x = df.drop("MEDV", axis=1)
y = df["MEDV"]

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LinearRegression

# %%
ln = LinearRegression()
ln.fit(X_train, y_train)

# %%
y_pred = ln.predict(X_test)

# %%
from sklearn.metrics import mean_squared_error, r2_score

# %%
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


