import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH = 60
IMG_HEIGHT = 60
NUM_CATEGORIES = 46
TEST_SIZE = 0.2


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, stupid = load_data((sys.argv[1]))
    # Split data into training and testing sets
    factorized, trash = pd.factorize(stupid)
    labels = tf.keras.utils.to_categorical(factorized)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    #
    # # Get a compiled neural network
    model = create_model()
    #
    # # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)
    #
    # # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)
    #
    # # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def without_small_signs(data_dir):
    counter = 1
    abs_path = os.path.abspath(data_dir)
    folders = os.listdir(data_dir)
    for folder in folders:
        path = os.path.join(abs_path, folder)
        pictures = os.listdir(path)
        for picture in pictures:
            img_path = os.path.join(path, picture)
            img = cv2.imread(img_path)
            width, height, dim = img.shape
            if width < 30 and height < 30:
                os.remove(img_path)
                counter += 1
    return counter


def color_change(data_dir):
    # parent_dir = "C:\\Users\\Team 3256_2\\PycharmProjects\\Practice\\traffic\\sorted_signs_gray"
    parent_dir = "C:\\Users\\jkim2\\Desktop\\CS50\\ezra\\traffic\\sorted_signs_gray"
    os.mkdir(parent_dir)
    folders = os.listdir(data_dir)
    for folder in folders:
        path = os.path.join(parent_dir, folder)
        os.mkdir(path)
        folder_path = os.path.join(data_dir, folder)
        pictures = os.listdir(folder_path)
        for picture in pictures:
            img_path = os.path.join(folder_path, picture)
            print(img_path)
            img = Image.open(img_path).convert('LA')
            pic_name = "gray_" + picture
            img.save(pic_name)
            gray_path = pic_name
            shutil.move(gray_path, path)
    return parent_dir

def sort_data(data_dir):
    """
    This sorts our data.
    """
    # Path to new directory with sorted traffic signs
    parent_dir = "C:\\Users\\jkim2\\Desktop\\CS50\\ezra\\traffic\\sorted_signs"
    os.mkdir(parent_dir)
    known_signs = []
    entries = os.listdir(data_dir)
    for picture in entries:
        New_Folder_Name = (picture.split('_'))[1]
        if New_Folder_Name not in known_signs:
            known_signs.append(New_Folder_Name)
            path = os.path.join(parent_dir, New_Folder_Name)
            os.mkdir(path)
        dir = "C:\\Users\\jkim2\\Desktop\\CS50\\ezra\\traffic\\cropped_signs"
        pic_path = os.path.join(dir,picture)
        path = os.path.join(parent_dir, New_Folder_Name)
        shutil.copy(pic_path, path)
    return parent_dir


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.

    data_dir = German Traffic Sign Recognition Benchmark
    """
    images = []
    labels = []
    entries = os.listdir(data_dir)
    for entry in entries:
        folder = os.path.join(data_dir, entry)
        file = os.listdir(folder)
        for image in file:
            path = os.path.join(folder, image)
            img = cv2.imread(path)
            if img is not None:
                dimensions = (IMG_WIDTH, IMG_HEIGHT)
                resized = cv2.resize(img, dimensions)
                # resized = cv2.cvtColor(np.float32(resized_pre), cv2.COLOR_BGR2GRAY)
                # resized_plus1 = np.expand_dims(blackAndWhiteImage, axis=2)
                images.append(resized)
                labels.append(entry)

    return((images, labels))
    # raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            16, (3,3), activation = 'relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.33),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
    # raise NotImplementedError

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            8, (3, 3), activation = 'relu', padding='valid', strides=(1,1), input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.Conv2D(
            16, (3, 3), activation = 'relu', padding='valid', strides=(1,1)
        ),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation = 'relu', padding='valid', strides=(1,1)
        ),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation = 'relu', padding='valid', strides=(1,1)
        ),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.UpSampling2D(size = (2,2)),

        tf.keras.layers.Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation = 'relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.UpSampling2D(size = (2,2)),

        tf.keras.layers.Conv2DTranspose(16, (3,3), padding='valid', strides=(1,1), activation = 'relu'),

        tf.keras.layers.Conv2DTranspose(1, (3,3), padding='valid', strides=(1,1), activation = 'relu'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")


    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
