import os
import json
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
import time

def partition(sizes, dataset, subset, i):
  if subset not in ["train", "test"]:
    raise KeyError(f"Subset {subset} does not exist!")

  for size in sizes:
    if size%10 != 0:
      raise ValueError(f"Dataset of size {size} does not allow for equal representation between classes!")

  imgs = {}
  dirs = os.listdir(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/{dataset}/{subset}/")
  dirs.remove("metadata.json")
  for dir in dirs:
    imgs[dir] = []
    for img in os.listdir(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/{dataset}/{subset}/{dir}/"):
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
        if os.path.exists(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}/{label}/"):
          os.system(f"rmdir D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}/{label}/")
          os.makedirs(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}/{label}/")
        else:
          os.makedirs(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}/{label}/")

        for img_name in datasets[size][label]:
          img = Image.open(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/{dataset}/{subset}/{label}/{img_name}")
          img.save(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}/{label}/{img_name}")
      else:
        if os.path.exists(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{subset}/{label}/"):
          os.system(f"rmdir D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{subset}/{label}/")
          os.makedirs(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{subset}/{label}/")
        else:
          os.makedirs(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{subset}/{label}/")

        for img_name in datasets[size][label]:
          img = Image.open(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/{dataset}/{subset}/{label}/{img_name}")
          img.save(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{subset}/{label}/{img_name}")

for i in range(15):
  partition([x for x in range(100, 10001, 100)], "mnist", "train", i)
  partition([1000], "mnist", "test", i)

  partition([x for x in range(100, 10001, 100)], "fashion_mnist", "train", i)
  partition([1000], "fashion_mnist", "test", i)

  partition([x for x in range(100, 10001, 100)], "sign_mnist", "train", i)
  partition([1000], "sign_mnist", "test", i)

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
  batch_size=batch_size, verbose=0)

  val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=validation_split,
  subset="validation",
  seed=seed,
  image_size=image_size,
  batch_size=batch_size, verbose=0)

  class_names = train_ds.class_names

  class_json = {}
  for i in range(len(class_names)):
    class_json[i] = class_names[i]

  with open(f"{dataset_path}/class_names.json", "w") as file:
    file.write(json.dumps(class_json))

  normalization_layer = tf.keras.layers.Rescaling(1./255)

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))

  first_image = image_batch[0]

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

def trainModel(dataset, dataset_path, size, epochs, i):
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
  with open("D:/Eamon/Documents/Coding/Python/SF/2024-2025/time.txt", "a") as file:
    file.write(f"Trial_{i+1}: {dataset}_{size}: {timer}\n")

  model.save(f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}.keras")

epochs = 100

for i in range(15):
  for dataset in ["mnist", "fashion_mnist", "sign_mnist"]:
    for size in [x for x in range(100, 10001, 100)]:
      print(f"Trial {i+1}: {dataset}_{size}")
      dataset_path = f"D:/Eamon/Documents/Coding/Python/SF/2024-2025/Trial_{i+1}/{dataset}_{size}"
      trainModel(dataset, dataset_path, size, epochs, i)
