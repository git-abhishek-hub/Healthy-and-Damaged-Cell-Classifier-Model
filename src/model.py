import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import(Conv2D, MaxPooling2D, Flatten,Dense, Dropout)

def build_model():
    model = Sequential()

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(64,64,3)
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    
    model.add(
        Conv2D( 64,(3,3), activation='relu')
    )
    model.add(
        MaxPooling2D( (2, 2))
        
    )
    model.add(
        Conv2D(128,(3,3), activation='relu')
    )
    model.add(
        MaxPooling2D((2,2))
    )
    model.add (
        Conv2D(256,(3,3), activation="relu")
    )
    model.add(
        MaxPooling2D((2,2))
    )
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
if __name__ == "__main__":
    model = build_model()
    model.summary()
    