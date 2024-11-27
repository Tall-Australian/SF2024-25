import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy

# Processing
with open("results.txt", "r") as file:
  data = {"mnist":{},"fashion_mnist":{}, "sign_mnist":{}}
  for line in file.readlines():
    line = line.replace(" ", "").replace("%", "")
    dataset, value = line.split(":")
    dataset, size = dataset.rsplit("_", 1)

    data[dataset][size] = value.replace("\n", "")
  print(data)

i = 1

for dataset in list(data.keys()):
  x = np.array([int(i) for i in list(data[dataset].keys())])
  y = np.array([float(i) for i in list(data[dataset].values())])

  # Plot raw data
  plt.subplot(len(list(data.keys())), 1, i)
  plt.plot(x, y, 'o')
  plt.xlabel("# of Images")
  plt.ylabel("Accuracy (%)")
  plt.title(f"Dataset size vs. Accuracy in {dataset}")

  # Plot linear regression
  reg = np.polyfit(x,y, deg=1)
  p = np.poly1d(reg)
  plt.plot(x,p(x), '--', label="Linear curve")
  r2 = r2_score(y, p(x))
  print(f"{dataset} linear r2 score: {r2}")

  # Plot logarithmic regression
  a = np.polyfit(np.log(x),y,deg=1)
  x_fitted = np.linspace(np.min(x), np.max(x), 100)
  y_fitted = a[0] * np.log(x_fitted) + a[1]
  plt.plot(x_fitted, y_fitted, '--k', label="Logarithmic curve")
  r2 = r2_score(y, y_fitted)
  print(f"{dataset} logarithmic r2 score: {r2}")

  i += 1
  plt.ylim(50,100)

plt.subplots_adjust(top=2,hspace=0.5)
plt.show()
