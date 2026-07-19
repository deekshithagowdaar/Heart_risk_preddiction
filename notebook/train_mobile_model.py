import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)
BATCH = 16

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7,1.3]
)

val_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(
    "dataset/mobile_images/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

val = val_gen.flow_from_directory(
    "dataset/mobile_images/val",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=10)

model.save("models/mobile_model.h5")