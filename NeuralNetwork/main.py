import os
import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers
import h5py
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from PyInstaller.utils.hooks import collect_submodules

assetPath = os.getcwd() + "\\NeuralNetwork\\"
print(assetPath)
print("\n\nNeural Network Utility\n----------------------\n")
print("TensorFlow version:", tf.__version__)
print("Numpy version:", np.__version__)
print("H5py version:", h5py.__version__)
print("Pillow version:", PIL.__version__, "\n")

print("All tensorflow warnings below can be ignored as they are irelevant to the execution of this program.\n")

def intTryParse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

def getImageFeatureFromPath(imagePath):
    image = Image.open(imagePath)
    data = np.asarray(image)

    x = data.shape[0]
    y = data.shape[1]

    if x > y:
        data = data[x-y:, 0:]
    if y > x:
        data = data[0:, y-x:]

    data = np.asarray(Image.fromarray(data).resize((64, 64)))
    print("Loaded asset: \"" + imagePath + "\" with resolution:", data.shape)
    return data

def load_dataset():
    assetPath = 'L:/Coding/BMA/NeuralNetwork/assets/'
    trainPath = assetPath + "train/"
    testPath = assetPath + "test/"
    trainClassNames = os.listdir(trainPath)
    testClassNames = os.listdir(testPath)
    totalClassNames = list(set(trainClassNames + testClassNames))
    totalClassNames.sort()
    classDictionary = {}
    
    inpt = ""
    if os.path.exists("dataset.h5py"):
        while inpt != "y" and inpt != "Y" and inpt != "n" and inpt != "N":
            inpt = input("Reload assets and rebuild dataset? (recommended only if assets directory was modified, can take a long time) [y/n]: ")
    else:
        inpt = "y"
    
    #recreate dataset file if requested
    if inpt == "y" or inpt == "Y":
        if os.path.exists("dataset.h5py"):
            os.remove("dataset.h5py")
        open("dataset.h5py", 'a').close()

    #get classes
    if inpt == "y" or inpt == "Y":
        print("\nClasses:")
    for i, className in enumerate(totalClassNames):
        classDictionary[className] = i
        if inpt == "y" or inpt == "Y":
            print("\"" + className + "\"")

    #read file
    f = h5py.File("dataset.h5py", "r+")

    if inpt == "y" or inpt == "Y":
        #load assets into datasets and save to dataset file
        train_set_x = f.create_dataset(name="train_set_x", shape=(0, 64, 64, 3), maxshape=(None, 64, 64, 3), dtype='i8', chunks=True, compression="gzip", compression_opts=1)
        train_set_y = f.create_dataset(name="train_set_y", shape=(0), maxshape=(None,), dtype='i8', chunks=True, compression="gzip", compression_opts=1)
        test_set_x = f.create_dataset(name="test_set_x", shape=(0, 64, 64, 3), maxshape=(None, 64, 64, 3), dtype='i8', chunks=True, compression="gzip", compression_opts=1)
        test_set_y = f.create_dataset(name="test_set_y", shape=(0), maxshape=(None,), dtype='i8', chunks=True, compression="gzip", compression_opts=1)

        print("\nAssets:")
        index = 0
        for trainClassName in trainClassNames:
            trainClassPath = trainPath + trainClassName + "/"
            for imagePath in os.listdir(trainClassPath):
                train_set_x.resize((index + 1, 64, 64, 3))
                train_set_y.resize((index + 1,))
                train_set_x[index] = getImageFeatureFromPath(trainClassPath + imagePath)
                train_set_y[index] = classDictionary[trainClassName]
                index += 1

        index = 0
        for testClassName in testClassNames:
            testClassPath = testPath + testClassName + "/"
            for imagePath in os.listdir(testClassPath):
                test_set_x.resize((index + 1, 64, 64, 3))
                test_set_y.resize((index + 1,))
                test_set_x[index] = getImageFeatureFromPath(testClassPath + imagePath)
                test_set_y[index] = classDictionary[testClassName]
                index += 1

        #repeat dataset until it contains around 1000 elements
        lenght = len(train_set_x)
        if(lenght < 1000):
            train_set_x.resize((1000, 64, 64, 3))
            train_set_y.resize((1000,))

            repeatCount = int(1000 / lenght)

            tf.repeat(train_set_x, repeatCount, axis=0)
            tf.repeat(train_set_y, repeatCount, axis=0)

        print("")
    else:
        #load datasets from dataset file
        train_set_x = f["train_set_x"]
        train_set_y = f["train_set_y"]
        test_set_x = f["test_set_x"]
        test_set_y = f["test_set_y"]

    print("Loaded dataset.\n")
    return train_set_x, train_set_y, test_set_x, test_set_y, classDictionary

def make_or_restore_model(inpt):
    checkpoints = ["checkpoints/" + name for name in os.listdir(f"{assetPath}checkpoints")]
    if checkpoints and (inpt == "n" or inpt == "N"):
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring model from", latest_checkpoint, "...\n")
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model...\n")

    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                        input_shape=(64,
                                    64,
                                    3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
    )

    m = keras.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(classDictionary))
        ])

    m.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return m

def predict(probability_model, testSetX, testSetY, yIsSet = True):
    predictions = probability_model.predict(testSetX)

    print("\nPredictions:\n")
    for i, prediction in enumerate(predictions):
        if yIsSet:
            print("Actual: \"" + list(classDictionary.keys())[list(classDictionary.values()).index(testSetY[i])] + "\"")
        else:
            print("Actual: \"" + testSetY + "\"")

        orderedPrediction = []
        for l, p in enumerate(prediction):
            orderedPrediction.append([l, round(p * 100, 3)])
        orderedPrediction = np.array(orderedPrediction)
        ind = np.argsort(orderedPrediction[:,1]); orderedPrediction = orderedPrediction[ind][::-1]
        for j, p in enumerate(orderedPrediction):
            name = list(classDictionary.keys())[list(classDictionary.values()).index(p[0])]
            if p[1] != 0:
                print("  " + str(j + 1) + ". \"" + name + "\" (" + str(p[1]) + "%)")
        if(i + 1 < len(predictions)):
            input("Press \"Enter\" for next image.")
        print("")

print("Loading dataset...")
train_set_x, train_set_y, test_set_x, test_set_y, classDictionary = load_dataset()

inpt = ""
if os.path.exists("checkpoints") and len(os.listdir('checkpoints')) > 0:
    while inpt != "y" and inpt != "Y" and inpt != "n" and inpt != "N":
        inpt = input("Do you want to create a new model? (needed if dataset was updated, else previous model will be used) [y/n]: ")
else:
    inpt = "y"

model = make_or_restore_model(inpt)

print("\nModel summary:\n")
model.summary()

if(inpt == "n" or inpt == "N"):
    inpt = ""
    while inpt != "1" and inpt != "2":
        inpt = input("\nDo you want to (1) train the model or (2) test the model [1/2]: ")
else:
    inpt = "1"

if inpt == "1":
    print("")
    epochInpt = ""
    while (not intTryParse(epochInpt)[1]) or int(epochInpt) < 1 or int(epochInpt) > 500:
        epochInpt = input("Epoch count [1 - 500]: ")
    epochs = int(epochInpt)


    print("\nBeginning to train model...\n")


    history = model.fit(train_set_x, train_set_y, epochs=epochs, shuffle='batch')
    print("\nFinished training model.\n")
    #model.save("checkpoints/model")

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(test_set_x, test_set_y, verbose=0)
    print("\nResults:")
    print("Loss:", test_loss)
    print("Accuracy:", test_acc, "\n")
    acc = history.history['accuracy']

    loss = history.history['loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss')
    plt.show()


    print("\nCreating probability (testing) model...")
    probability_model = keras.Sequential([
        model,
        layers.Softmax()
    ])

    print("\nModel summary:\n")
    probability_model.summary()

    inpt = ""
    while inpt != "1" and inpt != "2":
        inpt = input("\nDo you want to (1) test on images from the \"assets/test\" folder or (2) manually enter paths to images you want to test? [1/2]: ")

    if(inpt == "1"):
        print("\nPredicting on \"asset/test\" images:\n")
        predict(probability_model, test_set_x, test_set_y)
    else:
        print("")
        exitP = False
        while not exitP:
            inpt = ""
            while inpt != "1" and inpt != "2":
                inpt = input("Do you want to (1) test another image or (2) exit the program? [1/2]: ")

            if(inpt == "2"):
                exitP = True
            else:
                path = input("\nPath to image (not all images are supported and work): ")
                print("")

                open("temp.h5py", 'a').close()
                f2 = h5py.File("temp.h5py", "r+")

                if(os.path.exists(path)):
                    try:
                        image = getImageFeatureFromPath(path)
                        
                        if(image.shape == (64, 64, 3)):
                            print("\nPredicting on \"" + path + "\" image:\n")

                            xSet = f2.create_dataset(name="x", shape=(0, 64, 64, 3), maxshape=(None, 64, 64, 3), dtype='i8', chunks=True, compression="gzip", compression_opts=1)
                            xSet.resize((1, 64, 64, 3))
                            xSet[0] = image

                            predict(probability_model, xSet, path, False)
                        else:
                            print("Could not process image with path \"" + path + "\".")
                    except Exception:
                        print("\nCould not load or process image with path \"" + path + "\".")
                else:
                    print("File with path \"" + path + "\" does not exist.")

                f2.close()
                if(os.path.exists("temp.h5py")):
                    os.remove("temp.h5py")

            print("")

    del probability_model
    del model

print("Program ran to completion.")