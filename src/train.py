import numpy as np
import matplotlib.pyplot as plt

from model import build_model

def load_processed_dat():
    data = np.load(r"E:\Liver_on_chip\datas\new_processed_dataset.npz")
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test'] 
    y_test = data['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test

def train():
    print("Loading new processed dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dat()
    print("Building model...")
    
    model = build_model()
    print("Training model...")
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=16,
        batch_size=32
    )
    print("Training complete.")
    
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    
    model.save("livercell_classifier3.keras")
    print("Model saved as livercell_classifier.keras")
    
    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.show()
    
    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.show()
    
if __name__ == "__main__":

    train()