import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

def load_data(path="new_liver_dataset6K.npz"):
    
    data = np.load(path)
    images = data['images']
    labels = data['labels']
    return images, labels
def normalize(images):
    # Normalize pixel values to [0, 1]
    images = images.astype('float32') / 255.0
    return images

# def split_dataset(images,labels):
#     # Split the dataset into training and testing sets
#     X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, stratify=labels, random_state=42)
    
#     # Further split the training set into training and validation sets
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
#     return (
#     X_train,
#     X_val,
#     X_test,
#     y_train,
#     y_val,
#     y_test
# )    

def split_dataset(images, labels):

    indices = np.arange(len(images))
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    n = len(images)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_train = images[:train_end]
    y_train = labels[:train_end]

    X_val = images[train_end:val_end]
    y_val = labels[train_end:val_end]

    X_test = images[val_end:]
    y_test = labels[val_end:]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

def save_processed_dataset(
        save_path,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
):
    np.savez(
        save_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )
    
def preprocess():

    print("Loading dataset...")

    images, labels = load_data()

    print("Dataset Loaded")
    print(images.shape)
    print(labels.shape)

    print("\nNormalizing images...")

    images = normalize(images)

    print(
        f"Pixel Range: "
        f"{images.min():.3f} - "
        f"{images.max():.3f}"
    )

    print("\nSplitting dataset...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(
        images,
        labels
    )

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    save_processed_dataset(
        "new_processed_dataset6Kg.npz",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print("\nProcessed dataset saved.")
    
if __name__ == "__main__":
    preprocess()
    
