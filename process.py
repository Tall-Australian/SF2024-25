import pandas as pd
from tqdm import tqdm

def process_acc(j):
    # Processing
    with open(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{j+1}/results.txt", "r") as file:
        data = {"mnist":{},"fashion_mnist":{}, "sign_mnist":{}}
        for line in file.readlines():
            line = line.replace(" ", "").replace("%", "")
            dataset, value = line.split(":")
            dataset, size = dataset.rsplit("_", 1)

            data[dataset][size] = value.replace("\n", "")
        df = pd.DataFrame(data=data)
        df.to_csv(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{j+1}/results.csv")
    return data

acc_data = {}
for j in range(15):
    data = process_acc(j)
    acc_data[j+1] = data
    
def process_time():
    with open("D:/Eamon/Documents/Coding/Python/SF/2024-2025/time.txt", "r") as file:
        df = pd.DataFrame(columns=["trial","size","dataset","time"])
        print(df)
        for line in tqdm(file.readlines()):
            line = line.replace(" ", "")
            line_data = line.split(":")
            trial = int(line_data[0].split("_")[1])
            size = line_data[1].rsplit("_", 1)[1]
            dataset = line_data[1].rsplit("_", 1)[0]
            train_time = float(line_data[2].replace("\n", ""))
            
            data = {"trial": trial, "size":size, "dataset":dataset, "time":train_time}
            
            df2 = pd.DataFrame(data=data, index=[0])
            df = pd.concat([df, df2], ignore_index=True, axis=0)
        
        df = df.drop(df[df.duplicated(subset=["trial","size","dataset"])].index)
        print(df.shape)
        
        df.to_csv("D:/Eamon/Documents/Coding/Python/SF/2024-2025/time.csv")
        
        return df
    
df = process_time()
time_data = {}
for j in range(15):
    j += 1
    df2 = df[df["trial"] == j]
    data = {"mnist":{},"fashion_mnist":{}, "sign_mnist":{}}
    df2.apply(lambda x: data[x.loc["dataset"]].update({x.loc["size"]: x.loc["time"]}), axis=1)
    time_data[j] = data

mean_time_data = {"mnist":{},"fashion_mnist":{}, "sign_mnist":{}}

for size in range(100, 10001, 100):
    m_values = []
    f_values = []
    s_values = []
    for trial in range(15):
        m_values.append(time_data[trial+1]["mnist"][str(size)])
        f_values.append(time_data[trial+1]["fashion_mnist"][str(size)])
        s_values.append(time_data[trial+1]["sign_mnist"][str(size)])
    mean_time_data["mnist"][size] = sum(m_values)/len(m_values)
    mean_time_data["fashion_mnist"][size] = sum(f_values)/len(f_values)
    mean_time_data["sign_mnist"][size] = sum(s_values)/len(s_values)

# print(mean_time_data)
pd.Series(data=mean_time_data["mnist"]).to_csv("D:/Eamon/Documents/Coding/Python/SF/2024-2025/output/mean_time_mnist_data.csv", index=[0])
pd.Series(data=mean_time_data["fashion_mnist"]).to_csv("D:/Eamon/Documents/Coding/Python/SF/2024-2025/output/mean_time_fashion_mnist_data.csv", index=[0])
pd.Series(data=mean_time_data["sign_mnist"]).to_csv("D:/Eamon/Documents/Coding/Python/SF/2024-2025/output/mean_time_sign_mnist_data.csv", index=[0])

mean_acc_data = {"mnist":{},"fashion_mnist":{}, "sign_mnist":{}}

for size in range(100, 10001, 100):
    m_values = []
    f_values = []
    s_values = []
    for trial in range(15):
        m_values.append(float(acc_data[trial+1]["mnist"][str(size)]))
        f_values.append(float(acc_data[trial+1]["fashion_mnist"][str(size)]))
        s_values.append(float(acc_data[trial+1]["sign_mnist"][str(size)]))
    mean_acc_data["mnist"][size] = sum(m_values)/len(m_values)
    mean_acc_data["fashion_mnist"][size] = sum(f_values)/len(f_values)
    mean_acc_data["sign_mnist"][size] = sum(s_values)/len(s_values)

# print(mean_acc_data)
pd.Series(data=mean_acc_data["mnist"]).to_csv("D:/Eamon/Documents/Coding/Python/SF/2024-2025/output/mean_acc_mnist_data.csv", index=[0])
pd.Series(data=mean_acc_data["fashion_mnist"]).to_csv("D:/Eamon/Documents/Coding/Python/SF/2024-2025/output/mean_acc_fashion_mnist_data.csv", index=[0])
pd.Series(data=mean_acc_data["sign_mnist"]).to_csv("D:/Eamon/Documents/Coding/Python/SF/2024-2025/output/mean_acc_sign_mnist_data.csv", index=[0])

    