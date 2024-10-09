# %%
import pandas as pd

# %%
df = pd.read_csv("AirfoilSelfNoise.csv")

# %%
df.dropna(how="any", axis=1, inplace=True)

# %%
x = df.drop("SSPL", axis=1)
y = df["SSPL"]

# %%
df.info()

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
from sklearn.neighbors import KNeighborsRegressor

# %%
kn = KNeighborsRegressor(n_neighbors=3)
kn.fit(X_train, y_train)

# %%
y_pred = kn.predict(X_test)

# %%
from sklearn.metrics import mean_squared_error

# %%
mse1 = mean_squared_error(y_test, y_pred)

# %%
from sklearn.linear_model import LinearRegression

# %%
ln =  LinearRegression()
ln.fit(X_train, y_train)

# %%
y_pred = ln.predict(X_test)

# %%
result = "O Linear é melhor" if mean_squared_error(y_test, y_pred) > mse1 else "KNN é melhor"

# %%
print(result)

# %%



