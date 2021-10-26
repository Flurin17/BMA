import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import h5py
import numpy as np
import PIL
from PIL import Image

print("\n\nNeural Network Utility\n----------------------\n")
print("TensorFlow version:", tf.__version__)
print("Numpy version:", np.__version__)
print("H5py version:", h5py.__version__)
print("Pillow version:", PIL.__version__, "\n")

def getImageFeatureFromPath(imagePath):
    image = Image.open(imagePath)
    data = np.asarray(image)
    data[2] = data[2] / 255

    x = data.shape[0]
    y = data.shape[1]

    if x > y:
        data = data[x-y:, 0:]
    if y > x:
        data = data[0:, y-x:]

    data = np.asarray(Image.fromarray(data).resize((64, 64)))
    print("Loading asset: \"" + imagePath + "\" with resolution:", data.shape)
    return data

def load_dataset():
    maxImageLimit = 10000
    assetPath = "assets/"
    trainPath = assetPath + "train/"
    testPath = assetPath + "test/"
    trainClassNames = os.listdir(trainPath)
    testClassNames = os.listdir(testPath)
    totalClassNames = list(set(trainClassNames + testClassNames))
    classDictionary = {}
    
    #reset datasets by recreating dataset file
    if os.path.exists("dataset.h5py"):
        os.remove("dataset.h5py")
    open("dataset.h5py", 'a').close()

    f = h5py.File("dataset.h5py", "r+")
    train_set_x = f.create_dataset(name="train_set_x", shape=(maxImageLimit, 64, 64, 3), dtype='i8', chunks=True)
    train_set_y = f.create_dataset(name="train_set_y", shape=(maxImageLimit), dtype='i8', chunks=True)
    test_set_x = f.create_dataset(name="test_set_x", shape=(maxImageLimit, 64, 64, 3), dtype='i8', chunks=True)
    test_set_y = f.create_dataset(name="test_set_y", shape=(maxImageLimit), dtype='i8', chunks=True)

    print("\nClasses:")
    index = 0
    for className in totalClassNames:
        classDictionary[className] = index
        print("\"" + className + "\"")
        index += 1

    print("\nAssets:")
    index = 0
    for trainClassName in trainClassNames:
        trainClassPath = trainPath + trainClassName + "/"
        for imagePath in os.listdir(trainClassPath):
            train_set_x[index] = getImageFeatureFromPath(trainClassPath + imagePath)
            train_set_y[index] = classDictionary[className]
            index += 1
    train_set_x.resize((index + 1, 64, 64, 3))
    train_set_y.resize((index + 1,))

    index = 0
    for testClassName in testClassNames:
        testClassPath = testPath + testClassName + "/"
        for imagePath in os.listdir(testClassPath):
            test_set_x[index] = getImageFeatureFromPath(testClassPath + imagePath)
            test_set_y[index] = classDictionary[className]
            index += 1
    test_set_x.resize((index + 1, 64, 64, 3))
    test_set_y.resize((index + 1,))

    return train_set_x, train_set_y, test_set_x, test_set_y, classDictionary

def make_or_restore_model():
    checkpoints = ["checkpoints/" + name for name in os.listdir("checkpoints")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint, "...")
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model...")
    #m = keras.models.Sequential([
    #    layers.Flatten(input_shape=(64, 64, 3)),
    #    layers.Dense(256, activation='relu'),
    #    layers.Dropout(0.2),
    #    layers.Dense(len(classDictionary))
    #])
    m = keras.Sequential([
        layers.Flatten(input_shape=(64, 64, 3)),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(classDictionary))
    ])

    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=0.01, decay_steps=100000, decay_rate=0.96, staircase=True
    #)*/


    #m.compile(
    #    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    #    loss=keras.losses.SparseCategoricalCrossentropy(),
    #    metrics=[keras.metrics.SparseCategoricalAccuracy()]
    #)
    m.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return m

print("Loading dataset...")
train_set_x, train_set_y, test_set_x, test_set_y, classDictionary = load_dataset()

print("\nCreating model...")
model = make_or_restore_model()

print("\nModel summary:\n")
model.summary()

print("\nBeginning to train model...\n")
#history = model.fit(train_set_x, train_set_y, epochs=10, callbacks=[keras.callbacks.ModelCheckpoint(filepath="checkpoints/model-loss={loss:.3f}", save_best_only=True, monitor="loss", verbose=1, save_freq="epoch")], shuffle=False) # also try shuffle=batch
model.fit(train_set_x, train_set_y, epochs=10, callbacks=[keras.callbacks.ModelCheckpoint(filepath="checkpoints/model-loss={loss:.3f}", save_best_only=True, monitor="loss", verbose=1, save_freq="epoch")], shuffle=False)
print("\nFinished training model.\n")

print("\nEvaluating model...\n")
test_loss, test_acc = model.evaluate(test_set_x, test_set_y, verbose=2)

print("\nResults:\n")

print("Loss:", test_loss)
print("Accuracy:", test_acc)


print("\nPredicting on some test images:\n")
probability_model = keras.Sequential([
    model,
    layers.Softmax()
])
predictions = probability_model.predict(test_set_x)

index = 0
for prediction in predictions:
    first = np.argmax(prediction)
    firstName = list(classDictionary.keys())[list(classDictionary.values()).index(first)]
    firstProbability = prediction[first]
    prediction = np.delete(prediction, first)
    secondProbability = 0
    thirdProbability = 0
    second = np.argmax(prediction)
    secondName = list(classDictionary.keys())[list(classDictionary.values()).index(second)]
    secondProbability = prediction[second]
    prediction = np.delete(prediction, second)
    third = np.argmax(prediction)
    thirdName = list(classDictionary.keys())[list(classDictionary.values()).index(third)]
    thirdProbability = prediction[third]
    prediction = np.delete(prediction, third)
    fourth = np.argmax(prediction)
    fourthName = list(classDictionary.keys())[list(classDictionary.values()).index(fourth)]
    fourthProbability = prediction[fourth]
    prediction = np.delete(prediction, fourth)

    print("Actual: \"" + list(classDictionary.keys())[list(classDictionary.values()).index(test_set_y[index])] + "\"")
    print("  1. \"" + firstName + "\" (" + str(round(firstProbability * 100, 3)) + "%)")
    if secondProbability != 0:
        print("  2. \"" + secondName + "\" (" + str(round(secondProbability * 100, 3)) + "%)")
    if thirdProbability != 0:
        print("  3. \"" + thirdName + "\" (" + str(round(thirdProbability * 100, 3)) + "%)")
    if fourthProbability != 0:
        print("  4. \"" + fourthName + "\" (" + str(round(fourthProbability * 100, 3)) + "%)")
    print("")
    index += 1

del model