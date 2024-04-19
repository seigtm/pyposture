# Распознавание позы человека

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
```

Введем обозначение `keypoint` - это часть тела важная для определения позы человека. Ниже определим ярлыки с названием каждого `keypoint`.

```python
KEYPOINTS_QUANTITY = 17
MAX_PERSON_QUANTITY = 6
UNTRUSTABLE_KEYPOINT_THRESHOLD = 256 * 0.3
UNTRUSTABLE_PERSON_THRESHOLD = 256 * 0.3

keypoint_labels = [
    "nose", "left eye", "right eye", "left ear", "right ear", "left shoulder",
    "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
    "left hip", "right hip", "left knee", "right knee", "left ankle",
    "right ankle"
]
```

Загружаем предобученную модель `MOVENET` из локальной папки. Данная модель разработана компанией `Google` на основе фреймворка `Tensorflow`. На вход модель ожидает получить RGB изображение одного человека.

```python
model = hub.load("movenet")
movenet = model.signatures['serving_default']
```

Загрузим изображение из локальной папки и подготовим его для передачи входным параметром модели.

```python
image_path = '00030_00.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
X = tf.expand_dims(image, axis=0)
X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)

plt.title('Original Image')
plt.imshow(image)
```

```python
<matplotlib.image.AxesImage at 0x2d2fc03cdf0>
```

![png](simple-application-of-movenet_files/simple-application-of-movenet_7_1.png)

Получим предсказание для подготовленного ранее изображения. В качестве результата получим представление размера [1, 6, 56]. Расшифруем измерения:

- первое измерение равно количеству batch, для данной модели всегда равно 1, наличие данного измерения обоснованно особенностями Tensorflow.
- второе измерение равно максимальному количеству людей, которых модель может распознать на изображении.
- третье измерение представляет собой боксы/точки, описывающие части тела человека, и их оценку достоверности.

```python
movenet_result = movenet(X)['output_0'].numpy()
movenet_result.shape
predictions = movenet_result[0]
```

Так как мы знаем, что на изображении один человек, то нужно отобрать данные, достоверность, которых наибольшая. Последнее число в строке это достоверность всего предсказания.

```python
most_accurate_index = predictions[:, 55].argmax()
most_accurate_row = predictions[most_accurate_index, :]
most_accurate_row = (most_accurate_row * 256).astype(float)
```

В строке первое 51 значение связано с keypoint-ами, остальные 5 связаны с предсказанием в целом. Подготовим словарь, где для каждого keypoint будет указаны координаты и точность предсказания.

```python
keypoints_dict = {}
for i in range(KEYPOINTS_QUANTITY):
    keypoints_dict[keypoint_labels[i]] = {
        "x": most_accurate_row[i * 3],
        "y": most_accurate_row[i * 3 + 1],
        "confidence": most_accurate_row[i + 2]
    }
keypoints_dict
```

```python
{'nose': {'x': 53.68281936645508,
    'y': 110.08720397949219,
    'confidence': 205.25970458984375},
    'left eye': {'x': 48.38102340698242,
    'y': 116.32070922851562,
    'confidence': 48.38102340698242},
    'right eye': {'x': 49.89620590209961,
    'y': 103.71116638183594,
    'confidence': 116.32070922851562},
    'left ear': {'x': 55.47563934326172,
    'y': 126.15805053710938,
    'confidence': 159.2164764404297},
    'right ear': {'x': 59.88667678833008,
    'y': 98.42314910888672,
    'confidence': 49.89620590209961},
    'left shoulder': {'x': 85.4734878540039,
    'y': 148.44140625,
    'confidence': 103.71116638183594},
    'right shoulder': {'x': 98.42991638183594,
    'y': 87.9872055053711,
    'confidence': 196.3776092529297},
    'left elbow': {'x': 144.43655395507812,
    'y': 159.6914825439453,
    'confidence': 55.47563934326172},
    'right elbow': {'x': 154.67832946777344,
    'y': 79.3277359008789,
    'confidence': 126.15805053710938},
    'left wrist': {'x': 195.7560577392578,
    'y': 157.15045166015625,
    'confidence': 173.28790283203125},
    'right wrist': {'x': 201.97068786621094,
    'y': 73.93953704833984,
    'confidence': 59.88667678833008},
    'left hip': {'x': 178.46011352539062,
    'y': 136.68789672851562,
    'confidence': 98.42314910888672},
    'right hip': {'x': 177.13453674316406,
    'y': 94.58570098876953,
    'confidence': 194.20663452148438},
    'left knee': {'x': 251.61935424804688,
    'y': 149.62808227539062,
    'confidence': 85.4734878540039},
    'right knee': {'x': 250.86114501953125,
    'y': 95.4998779296875,
    'confidence': 148.44140625},
    'left ankle': {'x': 253.36509704589844,
    'y': 144.5836181640625,
    'confidence': 216.9861297607422},
    'right ankle': {'x': 249.85560607910156,
    'y': 103.63066101074219,
    'confidence': 98.42991638183594}}
```

В качестве визуализации будем отрисовывать три изображение: исходное, позу и позу наложенную на исходное

```python
connections = [('nose', 'left eye'), ('left eye', 'left ear'),
               ('nose', 'right eye'), ('right eye', 'right ear'),
               ('nose', 'left shoulder'), ('left shoulder', 'left elbow'),
               ('left elbow', 'left wrist'), ('nose', 'right shoulder'),
               ('right shoulder', 'right elbow'),
               ('right elbow', 'right wrist'), ('left shoulder', 'left hip'),
               ('right shoulder', 'right hip'), ('left hip', 'right hip'),
               ('left hip', 'left knee'), ('right hip', 'right knee')]


def draw_image(img):
    plt.subplot(1, 3, 1)
    plt.title('Person only')
    plt.axis('off')
    plt.imshow(img)


def draw_pose_only(img, keypoints_dict):
    plt.title('Pose only')
    plt.axis('off')
    plt.imshow((img / 255) / 255)
    for start_key, end_key in connections:
        if start_key in keypoints_dict and end_key in keypoints_dict:
            start_point = keypoints_dict[start_key]
            end_point = keypoints_dict[end_key]
            plt.plot([start_point["y"], end_point["y"]],
                     [start_point["x"], end_point["x"]],
                     linewidth=2)


def draw_pose_and_image(img, keypoints_dict, keypoint_labels):
    plt.title('Pose and person')
    plt.axis('off')
    plt.imshow(img)

    for i in range(KEYPOINTS_QUANTITY):
        if keypoints_dict[keypoint_labels[i]][
                "confidence"] > UNTRUSTABLE_KEYPOINT_THRESHOLD:
            plt.scatter(keypoints_dict[keypoint_labels[i]]["y"],
                        keypoints_dict[keypoint_labels[i]]["x"],
                        color='green')

    for start_key, end_key in connections:
        if start_key in keypoints_dict and end_key in keypoints_dict:
            start_point = keypoints_dict[start_key]
            end_point = keypoints_dict[end_key]
            plt.plot([start_point["y"], end_point["y"]],
                     [start_point["x"], end_point["x"]],
                     linewidth=2)


def visualize(img, keypoints_dict, keypoint_labels):
    plt.figure(figsize=(15, 5))
    draw_image(img)

    plt.subplot(1, 3, 3)
    draw_pose_and_image(img, keypoints_dict, keypoint_labels)

    plt.subplot(1, 3, 2)
    draw_pose_only(img, keypoints_dict)
```

Провизуализируем результаты предсказания на тестовом изображении

```python
img = tf.image.resize_with_pad(image, 256, 256)
img = tf.cast(img, dtype=tf.int32)
img = tf.expand_dims(img, axis=0)
img = img.numpy()[0]

visualize(img, keypoints_dict, keypoint_labels)
```

![png](simple-application-of-movenet_files/simple-application-of-movenet_17_0.png)

Получим предсказание для другой картинки, на которой будет несколько человек

```python
image_path = 'several_people.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
X = tf.expand_dims(image, axis=0)
X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)

plt.title('Original Image')
plt.imshow(image)

movenet_result = movenet(X)['output_0'].numpy()
movenet_result.shape
predictions = movenet_result[0]
```

![png](simple-application-of-movenet_files/simple-application-of-movenet_19_0.png)

Провизуализируем предсказание

```python
img = tf.image.resize_with_pad(image, 256, 256)
img = tf.cast(img, dtype=tf.int32)
img = tf.expand_dims(img, axis=0)
img = img.numpy()[0]
for i in range(MAX_PERSON_QUANTITY):
    row = (predictions[i, :] * 256).astype(float)
    if row[55] <= UNTRUSTABLE_PERSON_THRESHOLD:
        continue

    keypoints_dict = {}
    for i in range(KEYPOINTS_QUANTITY):
        keypoints_dict[keypoint_labels[i]] = {
            "x": row[i * 3],
            "y": row[i * 3 + 1],
            "confidence": row[i + 2]
        }

    visualize(img, keypoints_dict, keypoint_labels)
```

![png](simple-application-of-movenet_files/simple-application-of-movenet_21_0.png)

![png](simple-application-of-movenet_files/simple-application-of-movenet_21_1.png)

![png](simple-application-of-movenet_files/simple-application-of-movenet_21_2.png)

![png](simple-application-of-movenet_files/simple-application-of-movenet_21_3.png)
