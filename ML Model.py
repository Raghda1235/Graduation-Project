from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

# Define data directory path
data_dir = 'C:/Users/Kimo Store/Desktop/project/datasets4'

# Initialize lists to store filenames and their corresponding classes
file_paths = []
classes = []

# Iterate through each subdirectory (class) in the data directory
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    # Check if the subdirectory is a directory
    if os.path.isdir(class_dir):
        # Iterate through each file in the class directory
        for file_name in os.listdir(class_dir):
            # Append the file path and class label to the respective lists
            file_paths.append(os.path.join(class_dir, file_name))
            classes.append(class_name)

# Create a DataFrame to hold the file paths and their corresponding classes
data = {'filename': file_paths, 'class': classes}
df = pd.DataFrame(data)

# Perform random oversampling to handle class imbalance
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(df[['filename']], df['class'])

# Convert resampled data to flat arrays
X_resampled_flat = X_resampled['filename']

# Create a DataFrame from the resampled data
resampled_df = pd.DataFrame({'filename': X_resampled_flat, 'class': y_resampled})

# Split the resampled data into train and test sets
train_df, test_df = train_test_split(resampled_df, test_size=0.2, random_state=42)

# Split the training data into train and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Calculate the number of classes
num_classes = 2  # Number of classes in your dataset

# Define image dimensions
img_height, img_width = 224, 224
batch_size = 16

# Define data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define data augmentation for validation and test data (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data for training, validation, and test
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Ensure one-hot encoding for categorical cross-entropy loss
    shuffle=True  # Shuffle training data
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Path to save/load the model
model_path = 'trained_model.h5'

if os.path.exists(model_path):
    # Load the trained model
    model = load_model(model_path)
    print("Model loaded from disk.")
else:
    # Load the pre-trained MobileNet model without the top classification layer
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a dropout layer for regularization
    x = Dropout(0.5)(x)

    # Add a fully connected layer
    x = Dense(1024, activation='relu')(x)

    # Add a classification layer with softmax activation
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine the base model and custom layers
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model with a specific learning rate
    learning_rate = 0.001  # Specify the learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    # Save the trained model
    model.save(model_path)
    print("Model saved to disk.")

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    # Predict classes for test data
    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Get the class labels from the test generator
    class_labels = list(test_generator.class_indices.keys())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Generate a classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

# Function to capture an image from the webcam and preprocess it
def capture_and_preprocess_image():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was captured properly
        if not ret:
            print("Failed to capture image")
            continue

        # Display the captured frame
        cv2.imshow('Webcam', frame)

        # Press 'q' to capture the image and exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Resize the image to the target size
    img = cv2.resize(frame, (img_height, img_width))

    # Convert the image to array
    img = img_to_array(img)

    # Expand dimensions to match the input shape of the model
    img = np.expand_dims(img, axis=0)

    # Preprocess the image
    img = preprocess_input(img)

    return img

while True:
    # Capture and preprocess image from the webcam
    captured_image = capture_and_preprocess_image()

    # Predict the class of the captured image
    predicted_prob = model.predict(captured_image)
    predicted_class = np.argmax(predicted_prob, axis=1)[0]

    # Get the class labels
    class_labels = list(train_generator.class_indices.keys())

    # Display the captured image and the predicted class in the window title
    # Assume class_labels and predicted_class are already defined
    window_title = f'{class_labels[predicted_class]}'
    cv2.imshow(window_title, cv2.cvtColor(captured_image[0], cv2.COLOR_RGB2BGR))

    # Correct the print statement by concatenating the string with window_title
    print(f'Predicted Class: {window_title}')


    # Press 'x' to close the displayed image
    if cv2.waitKey(0) == ord('x'):
        cv2.destroyAllWindows()
        break