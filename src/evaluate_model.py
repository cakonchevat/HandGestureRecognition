import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

IMG_SIZE = 100

def load_labels(labels_path):
    with open(labels_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names, {name: idx for idx, name in enumerate(class_names)}

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(interpreter, input_details, output_details, image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return np.argmax(prediction)

def evaluate_model(data_dir, class_names, label_map, interpreter, input_details, output_details):
    y_true, y_pred = [], []
    for class_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            input_data = preprocess_image(img)
            pred_class = predict_image(interpreter, input_details, output_details, input_data)
            y_true.append(label_map[class_name])
            y_pred.append(pred_class)
    return y_true, y_pred

def display_results(y_true, y_pred, class_names):
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a TFLite model on a test dataset.")
    parser.add_argument('--model', type=str, default='trained_model/model_optimized.tflite', help='Path to TFLite model')
    parser.add_argument('--labels', type=str, default='trained_model/labels.txt', help='Path to labels.txt')
    parser.add_argument('--data', type=str, default='data', help='Path to test dataset')
    return parser.parse_args()

args = parse_arguments()
class_names, label_map = load_labels(args.labels)
interpreter, input_details, output_details = load_tflite_model(args.model)
y_true, y_pred = evaluate_model(args.data, class_names, label_map, interpreter, input_details, output_details)
display_results(y_true, y_pred, class_names)
