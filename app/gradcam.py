import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("models/image_model.h5")

last_conv_layer_name = "top_conv"

def get_gradcam_image(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    conv_outputs = conv_outputs * pooled_grads

    heatmap = np.mean(conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = (img_array[0] * 255).astype(np.uint8)

    superimposed = cv2.addWeighted(original, 0.65, heatmap, 0.35, 0)

    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)