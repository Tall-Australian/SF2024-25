import os
import json
from PIL import Image
import pandas as pd
import numpy as np
import datetime
import random
from tqdm import tqdm
import tensorflow as tf
import time

def proccessDataset(dataset_path, label, header):
  df = pd.read_csv(dataset_path, header=header)
  print(df.shape)
  data = pd.DataFrame(columns=["label", "img"])
  data["label"] = df[label]
  df = df.drop(label, axis=1)
  data["img"] = df.apply(lambda x: x.to_numpy().reshape(28,28), axis=1)
  return data

def genMetadata(classes, data):
  metadata = {}
  counts = data["label"].value_counts()
  for label in data["label"].unique():
    label = int(label)
    metadata[label] = {"name":classes[label],"count":int(counts[label])}
  print(metadata)
  return metadata

def saveImages(save_dir, data, metadata, tt):
  if os.path.exists(f"{save_dir}{tt}"):
    os.system(f"rmdir {save_dir}{tt}")
    os.makedirs(f"{save_dir}{tt}")
  else:
    os.makedirs(f"{save_dir}{tt}")

  for label in data["label"].unique():
    if not os.path.exists(f"{save_dir}{tt}{str(label)}/"):
      os.mkdir(f"{save_dir}{tt}{str(label)}/")

  data["img"] = data.apply(lambda x: Image.fromarray((x["img"] * 255).astype(np.uint8)), axis=1)
  data.apply(lambda x: x["img"].save(f"{save_dir}{tt}{str(x['label'])}/{str(datetime.datetime.now().timestamp()).replace('.','_')}.png"), axis=1)
  with open(f"{save_dir}{tt}metadata.json", "w") as file:
    file.write(json.dumps(metadata))

def setup(dataset_path, label, classes, save_dir, header):
  if "train" in dataset_path:
    tt = "train/"
  elif "test" in dataset_path:
    tt = "test/"
  else:
    exit()

  data = proccessDataset(dataset_path, label, header)
  metadata = genMetadata(classes, data)
  saveImages(save_dir, data, metadata, tt)

setup("./mnist_train.csv",0,["0","1","2","3","4","5","6","7","8","9"],"./mnist/",None)
setup("./mnist_test.csv",0,["0","1","2","3","4","5","6","7","8","9"],"./mnist/",None)
os.replace("./mnist_train.csv" "./mnist/mnist_train.csv")
os.replace("./mnist_test.csv" "./mnist/mnist_test.csv")

setup("./fashion_mnist_train.csv","label",["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"],"./fashion_mnist/",0)
setup("./fashion_mnist_test.csv","label",["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"],"./fashion_mnist/",0)
os.replace("./fashion_mnist_train.csv" "./fashion_mnist/fashion_mnist_train.csv")
os.replace("./fashion_mnist_test.csv" "./fashion_mnist/fashion_mnist_test.csv")


setup("./sign_mnist_train.csv","label",["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],"./sign_mnist/", 0)
setup("./sign_mnist_test.csv","label",["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],"./sign_mnist/", 0)
os.replace("./sign_mnist_train.csv", "./sign_mnist/sign_mnist_train.csv")
os.replace("./sign_mnist_test.csv", "./sign_mnist/sign_mnist_test.csv")

def partition(sizes, dataset, subset):
  if subset not in ["train", "test"]:
    raise KeyError(f"Subset {subset} does not exist!")

  for size in sizes:
    if size%10 != 0:
      raise ValueError(f"Dataset of size {size} does not allow for equal representation between classes!")

  imgs = {}
  dirs = os.listdir(f"./{dataset}/{subset}/")
  dirs.remove("metadata.json")
  for dir in dirs:
    imgs[dir] = []
    for img in os.listdir(f"./{dataset}/{subset}/{dir}/"):
      imgs[dir].append(img)

  datasets = {}
  for size in sizes:
    datasets[size] = {}
    for label in (pbar := tqdm(list(range(0,10)))):
      if subset == "train":
        pbar.set_description(f"Processing {dataset}_{size}")
      else:
        pbar.set_description(f"Processing {dataset}_{subset}")

      if dataset == "sign_mnist":
        if label == 4:
          label = 10
        elif label == 9:
          label = 11

      label = str(label)
      datasets[size][label] = random.sample(imgs[label], int(size / 10))

      if subset == "train":
        if os.path.exists(f"./{dataset}_{size}/{label}/"):
          os.system(f"rmdir ./{dataset}_{size}/{label}/")
          os.makedirs(f"./{dataset}_{size}/{label}/")
        else:
          os.makedirs(f"./{dataset}_{size}/{label}/")

        for img_name in datasets[size][label]:
          img = Image.open(f"./{dataset}/{subset}/{label}/{img_name}")
          img.save(f"./{dataset}_{size}/{label}/{img_name}")
      else:
        if os.path.exists(f"./{dataset}_{subset}/{label}/"):
          os.system(f"rmdir ./{dataset}_{subset}/{label}/")
          os.makedirs(f"./{dataset}_{subset}/{label}/")
        else:
          os.makedirs(f"./{dataset}_{subset}/{label}/")

        for img_name in datasets[size][label]:
          img = Image.open(f"./{dataset}/{subset}/{label}/{img_name}")
          img.save(f"./{dataset}_{subset}/{label}/{img_name}")

partition([100, 1000, 10000], "mnist", "train")
partition([1000], "mnist", "test")

partition([100, 1000, 10000], "fashion_mnist", "train")
partition([1000], "fashion_mnist", "test")

partition([100, 1000, 10000], "sign_mnist", "train")
partition([1000], "sign_mnist", "test")

def buildModel(dataset_path):
  dataset_url = dataset_path
  activation = "relu"
  batch_size = 10
  image_size = (28,28)
  seed = 123
  validation_split = 0.2
  num_classes = 10

  train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=validation_split,
  subset="training",
  seed=seed,
  image_size=image_size,
  batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=validation_split,
  subset="validation",
  seed=seed,
  image_size=image_size,
  batch_size=batch_size)

  class_names = train_ds.class_names
  print(class_names)

  class_json = {}
  for i in range(len(class_names)):
    class_json[i] = class_names[i]

  with open(f"{dataset_path}/class_names.json", "w") as file:
    file.write(json.dumps(class_json))

  normalization_layer = tf.keras.layers.Rescaling(1./255)

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))

  first_image = image_batch[0]
  print(np.min(first_image), np.max(first_image))

  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    tpu = True
    strategy = tf.distribute.TPUStrategy(resolver)
  except ValueError:
    strategy = tf.distribute.get_strategy()
    tpu = False

  device_name = tf.test.gpu_device_name() if len(tf.test.gpu_device_name()) > 0 else "/CPU:0"

  with strategy.scope() if tpu else tf.device(device_name):
    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(32, 3, activation=activation),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation=activation),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation=activation),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(4096, activation=activation),
      tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

  model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

  return train_ds, val_ds, model

# SET IMPORTANT SETTINGS HERE!!!

dataset = "mnist"
size = 10000
dataset_path = f"./{dataset}_{size}"
epochs = 100

# SET IMPORTANT SETTINGS HERE!!!

train_ds, val_ds, model = buildModel(dataset_path)

start = time.time()

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  verbose=0
)

end = time.time()
timer = end - start
print(f"The {size} image {dataset} model took: {timer} seconds")

model.save(f"{dataset}_{size}.keras")
model.summary()
model.evaluate(train_ds)
model.evaluate(val_ds)

model = tf.keras.models.load_model(f"{dataset}_{size}.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
  f"./{dataset}_test/",
  image_size=(28,28))

results = model.evaluate(test_ds)
print("test loss, test acc:", results)

def predict(model, image_path, image_class, class_names, verbose=False):
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

image_data = pd.DataFrame(columns=["image_name","image_class","image_path"])
for dir in os.listdir(f"./{dataset}_test/"):
  for image in os.listdir(f"./{dataset}_test/{dir}/"):
    data = {
        "image_name": f"./{dataset}_test/{dir}/{image.replace('.png','')}",
        "image_class": dir,
        "image_path": f"./{dataset}_test/{dir}/{image}"
    }
    df = pd.DataFrame(data,index=[0])
    image_data = pd.concat([image_data, df],ignore_index=True)

with open(f"./{dataset}_{size}/class_names.json", "r") as file:
    class_names = json.load(file)

results = pd.DataFrame(columns=["raw","confidence","prediction","class","accuracy"])

print(image_data.shape)

for index, row in tqdm(image_data.iterrows(), total=len(image_data.index)):
  # print(row)
  pred = predict(model,row["image_path"],row["image_class"], class_names)
  results = pd.concat([results,pred], ignore_index=True)

print(results.shape)
results.to_csv(f"./{dataset}_{size}_results.csv")

results

print((results["accuracy"].value_counts()[1] / 1000) * 100)

from matplotlib import pyplot as plt
import seaborn as sns
results.groupby('accuracy').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)