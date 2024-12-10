import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd

def lin_func(x, y, meta):
    reg = np.polyfit(x,y,deg=1)
    p = np.poly1d(reg)
    r2 = r2_score(y, p(x))
    print(f"{meta["name"]} linear r2 score: {r2}")
    return r2, x, p(x)

def log_func(x, y, meta):
    a = np.polyfit(np.log(x),y,deg=1)
    x_fitted = np.linspace(np.min(x), np.max(x), 100)
    y_fitted = a[0] * np.log(x_fitted) + a[1]
    r2 = r2_score(y, y_fitted)
    print(f"{meta["name"]} logarithmic r2 score: {r2}")
    return r2, x_fitted, y_fitted

def sqrt_func(x, y, meta):
    coeffs = np.polyfit(np.sqrt(x),y,1)
    x_fitted = np.linspace(min(x),max(x),100)
    y_fitted = coeffs[0]*np.sqrt(x_fitted)+coeffs[1]
    r2 = r2_score(y, y_fitted)
    print(f"{meta["name"]} square root r2 score: {r2}")
    return r2, x_fitted, y_fitted

def exp_func(x, y, meta):
    coeffs = np.polyfit(x, np.log(y), 1)
    x_fitted = np.linspace(min(x),max(x),100)
    y_fitted = np.exp(coeffs[1])*np.exp(coeffs[0]*x_fitted)
    r2 = r2_score(y, y_fitted)
    print(f"{meta["name"]} exponential r2 score: {r2}")
    return r2, x_fitted, y_fitted

def plot(dataset, meta={"name":"MNIST","metric":"Accuracy (%)","id":"acc"}):
    df = pd.read_csv(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/output/mean_{meta["id"]}_{dataset}_data.csv", index_col=0)
    print(df)
    
    x = np.array(df.index)
    y = np.array(df["0"])
        
    # Plot raw data
    plt.plot(x, y, 'o')
    plt.xlabel("# of Images")
    plt.ylabel(meta["metric"])
    plt.title(f"Dataset Size vs. {meta["metric"]} in {meta["name"]}")

    r2_values = []
    functions = ["Linear", "Logarithmic", "Square Root", "Exponential"]
    
    # Plot linear regression
    r2,x_fit,y_fit = lin_func(x,y,meta)
    plt.plot(x_fit, y_fit, '--', label="Linear function")
    r2_values.append(r2)

    # Plot logarithmic regression
    r2,x_fit,y_fit = log_func(x,y,meta)
    plt.plot(x_fit, y_fit, '--k', label="Logarithmic function")
    r2_values.append(r2)
    
    # Plot square root regression
    r2,x_fit,y_fit = sqrt_func(x,y,meta)
    plt.plot(x_fit, y_fit, '--g', label="Square root function")
    r2_values.append(r2)
    
    # Plot exponential regression
    r2,x_fit,y_fit = exp_func(x,y,meta)
    plt.plot(x_fit, y_fit, '--m', label="Exponential function")
    r2_values.append(r2)
    
    print(f"Best Function: {functions[r2_values.index(max(r2_values))]} - {max(r2_values)}")

    plt.xlim(100,10000)
    if meta["id"] == "acc":
        plt.ylim(50,100)
    elif meta["id"] == "time":
        plt.ylim(0,600)

    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

for dataset in ["mnist", "fashion_mnist", "sign_mnist"]:
    plot(dataset, meta={"name":dataset.upper(),"metric":"Accuracy (%)","id":"acc"})

for dataset in ["mnist", "fashion_mnist", "sign_mnist"]:
    plot(dataset, meta={"name":dataset.upper(),"metric":"Time (Seconds)","id":"time"})
