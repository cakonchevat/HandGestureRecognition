import os
import cv2
import numpy as np
from keras._tf_keras import keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = 100
DATA_DIR = "data"
EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
KFOLD_SPLITS = 5


def load_and_augment_images(data_dir):
    X, y = [], []
    class_names = sorted(os.listdir(data_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        folder = os.path.join(data_dir, class_name)
        label = label_map[class_name]
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

            flipped = cv2.flip(img, 1)
            bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
            rotated = cv2.warpAffine(img, cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), 15, 1.0),
                                     (IMG_SIZE, IMG_SIZE))
            noise = img + np.random.normal(0, 10, img.shape).astype(np.uint8)
            X.extend([flipped, bright, rotated, noise])
            y.extend([label] * 4)

    return np.array(X), np.array(y), class_names


def preprocess_data(X, y, num_classes):
    X = X.astype('float32') / 255.0
    y = keras.utils.to_categorical(y, num_classes=num_classes)
    return X, y


def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Training History')
    plt.show()


def plot_confusion_matrix(model, X_val, y_val, class_names):
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict(X_val), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    print(classification_report(y_true, y_pred, target_names=class_names))


def export_model(model, class_names):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    os.makedirs("trained_model", exist_ok=True)
    with open("trained_model/model_optimized.tflite", "wb") as f:
        f.write(tflite_model)
    with open("trained_model/labels.txt", "w") as f:
        for label in class_names:
            f.write(label + "\n")


X, y, class_names = load_and_augment_images(DATA_DIR)
X, y = preprocess_data(X, y, num_classes=len(class_names))

# Use 20% as test set
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42)

kf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
y_int_main = np.argmax(y_main, axis=1)

best_val_acc = 0
best_model_path = ""

for fold, (train_idx, val_idx) in enumerate(kf.split(X_main, y_int_main)):
    print(f"\n--- Fold {fold + 1}/{KFOLD_SPLITS} ---")
    X_train, X_val = X_main[train_idx], X_main[val_idx]
    y_train, y_val = y_main[train_idx], y_main[val_idx]

    model = build_cnn_model((IMG_SIZE, IMG_SIZE, 3), len(class_names))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = f"best_model_fold{fold + 1}.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    val_acc = max(history.history['val_accuracy'])
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = checkpoint_path

    plot_training_history(history)
    plot_confusion_matrix(model, X_val, y_val, class_names)

print(f"\nEvaluating best model from fold on final test set: {best_model_path}")
best_model = load_model(best_model_path)
plot_confusion_matrix(best_model, X_test, y_test, class_names)
export_model(best_model, class_names)
