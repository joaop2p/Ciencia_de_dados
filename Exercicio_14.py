# %%
import pandas as pd

# %%
df = pd.read_csv("iris.csv")

# %%
df.dropna(how="any", axis=1, inplace=True)

# %%
df

# %%
df.drop("species", axis=1, inplace=True)

# %%
df.info()

# %%
from sklearn.cluster import KMeans

# %%
km = KMeans(random_state=42)

# %%
km.fit(df)

# %%
cl = km.predict(df)

# %%
from sklearn.metrics import silhouette_score

# %%
inert = km.inertia_
sl = silhouette_score(df, cl)

# %%
print(inert, sl)


