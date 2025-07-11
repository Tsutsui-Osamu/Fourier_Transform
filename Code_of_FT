import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files



# CSV読み込み
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
x = df.iloc[:, 0].tolist()
y = df.iloc[:, 1].tolist()

# フーリエ変換
class FT_calc:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def calculation(self, w):
        sum_cos = 0
        sum_sin = 0
        for i in range(len(self.x) - 1):
            dx = self.x[i+1] - self.x[i]
            sum_cos += dx * self.y[i] * np.cos(w * self.x[i])
            sum_sin += dx * self.y[i] * np.sin(w * self.x[i])
        return np.sqrt(sum_cos**2 + sum_sin**2)

# 計算とプロット
ft = FT_calc(x, y)
w_values = np.arange(0, 200, 0.1)
sum_results = [ft.calculation(w) for w in w_values]

plt.plot(w_values, sum_results)
plt.xlabel('Frequency (w)')
plt.ylabel('magnitude')
plt.show()
