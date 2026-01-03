import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def apply_heatmap(img_path, heatmap, intensity=0.6):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(heatmap_color, intensity, img, 1 - intensity, 0)

    cv2.imwrite("gradcam_output.png", superimposed_img)
    print("Saved: gradcam_output.png")

def generate_gradcam(img_path):
    model = tf.keras.models.load_model("models/dense_best.h5")

    last_conv_layer = "conv5_block16_concat"

    img_array = preprocess_image(img_path)

    heatmap = make_gradcam_heatmap(model, img_array, last_conv_layer)

    apply_heatmap(img_path, heatmap)

if __name__ == "__main__":
    test_img = "test_images/ti1.jpg"
    generate_gradcam(test_img)
