import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import h5py
import numpy as np
import PIL
from PIL import Image

#TODO
#disable tensorflow errors and warnings (does not work yet)
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

print("\n\nNeural Network Utility\n----------------------\n")
print("TensorFlow version:", tf.__version__)
print("Numpy version:", np.__version__)
print("H5py version:", h5py.__version__)
print("Pillow version:", PIL.__version__, "\n")

def getImageFeatureFromPath(imagePath):
    image = Image.open(imagePath)
    data = np.asarray(image)
    data[2] = data[2] / 255 #normalize values

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
    
    #TODO: option to just load dataset instead of reload everythings
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
    m = keras.Sequential([
        layers.Flatten(input_shape=(64, 64, 3)),
        layers.Dense(512, activation='relu'),
        layers.Dense(len(classDictionary))
    ])

    m.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-7),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return m

print("Loading dataset...")
train_set_x, train_set_y, test_set_x, test_set_y, classDictionary = load_dataset()

print("\nCreating model...")
model = make_or_restore_model()

print("\nModel summary:\n")
model.summary()

#TODO ask user
epochs = 2
steps_per_epoch = 5

print("\nBeginning to train model...\n")#TODO progress logging (to csv?)
model.fit(train_set_x, train_set_y, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[keras.callbacks.ModelCheckpoint(filepath="checkpoints/model-loss={loss:.3f}", save_best_only=True, monitor="loss", verbose=0, save_freq="epoch")], shuffle='batch')
print("\nFinished training model.\n")

print("Evaluating model...")
test_loss, test_acc = model.evaluate(test_set_x, test_set_y, verbose=0)
del model

print("\nResults:")

print("Loss:", test_loss)
print("Accuracy:", test_acc)

print("\nCreating probability model...")
probability_model = keras.Sequential([
    make_or_restore_model(),
    layers.Softmax()
])

print("\nModel summary:\n")
probability_model.summary()

print("\nPredicting on some test images:\n")
predictions = probability_model.predict(test_set_x)

for i, prediction in enumerate(predictions):
    print("Actual: \"" + list(classDictionary.keys())[list(classDictionary.values()).index(test_set_y[i])] + "\"")
    orderedPrediction = []
    for l, p in enumerate(prediction):
        orderedPrediction.append([l, round(p * 100, 3)])
    orderedPrediction = np.array(orderedPrediction)
    ind = np.argsort(orderedPrediction[:,1]); orderedPrediction = orderedPrediction[ind][::-1]
    for j, p in enumerate(orderedPrediction):
        name = list(classDictionary.keys())[list(classDictionary.values()).index(p[0])]
        if p[1] != 0:
            print("  " + str(j + 1) + ". \"" + name + "\" (" + str(p[1]) + "%)")

del probability_model