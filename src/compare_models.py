import numpy as np
import tensorflow as tf
import cv2
import os

MODEL_PATH_1 = "trained_model/model_optimized.tflite"  # Мој модел
MODEL_PATH_2 = "trained_model/model_unquant.tflite"  # Teachable Machine модел

LABELS_PATH = "trained_model/labels.txt"
IMG_SIZE = 300
TEST_IMAGES_DIR = "test_images"

def load_labels():
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_img = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_img = np.expand_dims(input_img.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

def evaluate_models():
    labels = load_labels()
    model1 = load_tflite_model(MODEL_PATH_1)
    model2 = load_tflite_model(MODEL_PATH_2)

    for filename in os.listdir(TEST_IMAGES_DIR):
        if not filename.endswith((".png", ".jpg")):
            continue
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, filename))
        prediction1 = predict(model1, img)
        prediction2 = predict(model2, img)

        label1 = labels[np.argmax(prediction1)]
        label2 = labels[np.argmax(prediction2)]
        print(f"Image: {filename}")
        print(f" - Model 1 prediction: {label1} ({np.max(prediction1):.2f})")
        print(f" - Model 2 prediction: {label2} ({np.max(prediction2):.2f})\n")

if __name__ == "__main__":
    evaluate_models()
