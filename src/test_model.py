import cv2
import numpy as np
import tensorflow as tf
import math
from cvzone.HandTrackingModule import HandDetector

MODEL_PATH = "trained_model/model_optimized.tflite"
LABELS_PATH = "trained_model/labels.txt"
IMG_SIZE = 300
OFFSET = 20
CONFIDENCE_THRESHOLD = 0.65


def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def load_labels():
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_hand(img, bbox):
    x, y, w, h = bbox
    img_crop = img[max(0, y - OFFSET): y + h + OFFSET, max(0, x - OFFSET): x + w + OFFSET]

    if img_crop.size == 0:
        return None

    aspect_ratio = h / w
    white_img = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

    try:
        if aspect_ratio > 1:
            k = IMG_SIZE / h
            new_w = math.ceil(k * w)
            resized = cv2.resize(img_crop, (new_w, IMG_SIZE))
            gap = math.ceil((IMG_SIZE - new_w) / 2)
            white_img[:, gap:gap + new_w] = resized
        else:
            k = IMG_SIZE / w
            new_h = math.ceil(k * h)
            resized = cv2.resize(img_crop, (IMG_SIZE, new_h))
            gap = math.ceil((IMG_SIZE - new_h) / 2)
            white_img[gap:gap + new_h, :] = resized
        return white_img, img_crop
    except Exception as e:
        print("Грешка при промена на димензиите:", e)
        return None


def predict(interpreter, input_details, output_details, image):
    input_img = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_img = np.expand_dims(input_img.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction


def draw_prediction(img, label, confidence, x, y, w, h):
    if confidence >= CONFIDENCE_THRESHOLD:
        color = (128, 0, 128)  # purple

        cv2.rectangle(img, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), color, 4)

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
        label_width, label_height = label_size
        label_x = x - OFFSET
        label_y = y - OFFSET - 10

        cv2.rectangle(
            img,
            (label_x - 5, label_y - label_height - 10),
            (label_x + label_width + 5, label_y + 5),
            color,
            cv2.FILLED
        )

        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),  # white text
            1
        )


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    interpreter, input_details, output_details = load_model()
    labels = load_labels()

    while True:
        success, frame = cap.read()
        if not success:
            continue

        output_img = frame.copy()
        hands, _ = detector.findHands(frame)

        if hands:
            hand = hands[0]
            bbox = hand['bbox']
            result = preprocess_hand(frame, bbox)

            if result:
                white_img, img_crop = result
                prediction = predict(interpreter, input_details, output_details, white_img)
                index = int(np.argmax(prediction))
                confidence = prediction[index]
                label = labels[index]
                draw_prediction(output_img, label, confidence, bbox[0], bbox[1], bbox[2], bbox[3])

        cv2.imshow("Image", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()