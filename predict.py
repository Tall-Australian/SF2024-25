import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd

def predict(model, image_path, image_class, class_names, i, verbose=False):
  image = tf.keras.preprocessing.image.load_img(image_path, target_size=(28,28))
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = np.expand_dims(image, axis = 0)

  prediction = model.predict(image, verbose=verbose)
  confidence = tf.nn.softmax(prediction[0]).numpy() * 100
  predicted_class = np.argmax(prediction, axis=1)

  if verbose:
    print(image_path)
    print(f"Raw prediction: {prediction}")
    print(f"Prediction confidence: {confidence}")
    print(f"Class prediction: {class_names[str(predicted_class[0])]}")
    print(f"Actual class: {image_class}")
    print(f"Result: {1 if class_names[str(predicted_class[0])]==image_class else 0}")

  data = {
      "raw": [prediction.flatten().tolist()],
      "confidence": [confidence.flatten().tolist()],
      "prediction": class_names[str(predicted_class[0])],
      "class": image_class,
      "accuracy": 1 if class_names[str(predicted_class[0])]==image_class else 0
  }

  df = pd.DataFrame(data, columns=list(data.keys()), index=[0])
  return df

def test(model, dataset, size, i):
    image_data = pd.DataFrame(columns=["image_name","image_class","image_path"])
    for dir in os.listdir(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_test/"):
        for image in os.listdir(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_test/{dir}/"):
            data = {
                "image_name": f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_test/{dir}/{image.replace('.png','')}",
                "image_class": dir,
                "image_path": f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_test/{dir}/{image}"
            }
            df = pd.DataFrame(data,index=[0])
            image_data = pd.concat([image_data, df],ignore_index=True)

    with open(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/{dataset}_class_names.json", "r") as file:
        class_names = json.load(file)

    results = pd.DataFrame(columns=["raw","confidence","prediction","class","accuracy"])

    # print(image_data.shape)

    for index, row in tqdm(image_data.iterrows(), total=len(image_data.index)):
        # print(row)
        pred = predict(model,row["image_path"],row["image_class"], class_names, i)
        results = pd.concat([results,pred], ignore_index=True)

    # print(results.shape)
    print(f"{dataset}_{size}: {results['accuracy'].value_counts()[1] / 1000 * 100}%")
    with open(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/results.txt", "a") as file:
        file.write(f"{dataset}_{size}: {results['accuracy'].value_counts()[1] / 1000 * 100}%\n")
    
    results.to_csv(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/results/Trial_{i+1}_{dataset}_{size}_results.csv")
    
    return results

for i in range(1):
  i = 2
  for dataset in ["mnist", "fashion_mnist", "sign_mnist"]:
    data = []
    for size in [x for x in range(100, 10001, 100)]:
      model = tf.keras.models.load_model(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}.keras")
      test_ds = tf.keras.utils.image_dataset_from_directory(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_test/", image_size=(28,28))
    
      data.append(test(model, dataset, size, i))
    
    df = pd.concat(data, ignore_index=True)
    df.to_csv(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_results.csv")