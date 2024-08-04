import numpy as np
import pandas as pd

array_2d = np.random.rand(3000,88)
df = pd.DataFrame(array_2d)
df.to_csv('array_2d.csv', index=False, header=False)
