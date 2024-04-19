import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub

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
model = hub.load("movenet")
movenet = model.signatures['serving_default']

image_path = '00030_00.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
X = tf.expand_dims(image, axis=0)
X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)

plt.title('Original Image')
plt.imshow(image)
plt.show()  # Ensure the plot is displayed

movenet_result = movenet(X)['output_0'].numpy()
movenet_result.shape
predictions = movenet_result[0]

most_accurate_index = predictions[:, 55].argmax()
most_accurate_row = predictions[most_accurate_index, :]
most_accurate_row = (most_accurate_row * 256).astype(float)

keypoints_dict = {}
for i in range(KEYPOINTS_QUANTITY):
    keypoints_dict[keypoint_labels[i]] = {
        "x": most_accurate_row[i * 3],
        "y": most_accurate_row[i * 3 + 1],
        "confidence": most_accurate_row[i + 2]
    }
print(keypoints_dict)

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


def visualize(img, keypoints_dict, keypoint_labels, description=''):
    plt.figure(figsize=(15, 5))
    draw_image(img)

    plt.subplot(1, 3, 3)
    draw_pose_and_image(img, keypoints_dict, keypoint_labels)

    plt.subplot(1, 3, 2)
    draw_pose_only(img, keypoints_dict)

    plt.savefig(f'result_{description}.png')
    plt.show()  # Ensure the plot is displayed

img = tf.image.resize_with_pad(image, 256, 256)
img = tf.cast(img, dtype=tf.int32)
img = tf.expand_dims(img, axis=0)
img = img.numpy()[0]

visualize(img, keypoints_dict, keypoint_labels, 'single_person')

image_path = 'several_people.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
X = tf.expand_dims(image, axis=0)
X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)

plt.title('Original Image')
plt.imshow(image)
plt.show()  # Ensure the plot is displayed

movenet_result = movenet(X)['output_0'].numpy()
movenet_result.shape
predictions = movenet_result[0]

img = tf.image.resize_with_pad(image, 256, 256)
img = tf.cast(img, dtype=tf.int32)
img = tf.expand_dims(img, axis=0)
img = img.numpy()[0]
for i in range(MAX_PERSON_QUANTITY):
    row = (predictions[i, :] * 256).astype(float)
    if row[55] <= UNTRUSTABLE_PERSON_THRESHOLD:
        continue

    keypoints_dict = {}
    for j in range(KEYPOINTS_QUANTITY):
        keypoints_dict[keypoint_labels[j]] = {
            "x": row[j * 3],
            "y": row[j * 3 + 1],
            "confidence": row[j + 2]
        }

    visualize(img, keypoints_dict, keypoint_labels, f'multiple_persons_{i}')
