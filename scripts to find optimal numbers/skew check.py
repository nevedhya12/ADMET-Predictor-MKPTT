import numpy as np, pandas as pd
y = np.load("scaffold_dataset/Excretion_y.npy")
print("skew:", pd.Series(y).skew())
print(np.percentile(y, [0, 25, 50, 75, 95, 100]))