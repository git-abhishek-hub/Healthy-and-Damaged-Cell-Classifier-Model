import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# ====================================================
# 1. LOAD DATA
# ====================================================

data = np.load(r"new_processed_dataset6K.npz")

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

print("Original Shapes")
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# ====================================================
# 2. RESIZE IMAGES TO 224×224
# ====================================================

def resize_images(images):
    resized = []

    for img in images:

        # Convert normalized images back to uint8
        img = (img * 255).astype(np.uint8)

        img = cv2.resize(img, (224, 224))

        resized.append(img)

    return np.array(resized)


X_train = resize_images(X_train)
X_val = resize_images(X_val)
X_test = resize_images(X_test)


print("\nAfter Resizing")
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# ====================================================
# 3. RESNET PREPROCESSING
# ====================================================

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)


# ====================================================
# 4. LOAD RESNET50
# ====================================================

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)


# ====================================================
# 5. FREEZE RESNET LAYERS
# ====================================================

base_model.trainable = False


# ====================================================
# 6. BUILD MODEL
# ====================================================

model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation='sigmoid'
    )

])


# ====================================================
# 7. COMPILE MODEL
# ====================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ====================================================
# 8. MODEL SUMMARY
# ====================================================

model.summary()


# ====================================================
# 9. TRAIN MODEL
# ====================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=10,

    batch_size=16,

    verbose=1
)


# ====================================================
# 10. EVALUATE
# ====================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)


# ====================================================
# 11. PREDICTIONS
# ====================================================

y_pred_prob = model.predict(X_test)

y_pred = (
    y_pred_prob > 0.5
).astype(int)


# ====================================================
# 12. CLASSIFICATION REPORT
# ====================================================

print("\nClassification Report")

print(

    classification_report(

        y_test,
        y_pred,

        target_names=[
            "Healthy",
            "Damaged"
        ]
    )
)


# ====================================================
# 13. CONFUSION MATRIX
# ====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")

print(cm)


# ====================================================
# 14. ACCURACY CURVE
# ====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.title("ResNet50 Accuracy")

plt.show()


# ====================================================
# 15. LOSS CURVE
# ====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.title("ResNet50 Loss")

plt.show()


# ====================================================
# 16. SAVE MODEL
# ====================================================

model.save(
    "resnet50_liver.keras"
)

print("\nModel Saved Successfully!")