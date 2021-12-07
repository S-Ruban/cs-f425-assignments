import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
import seaborn as sns


class CNN_Model:
    def __init__(
        self,
        layers_info,
        input_shape,
        num_classes,
        model_num,
        weights_folder="./weights/",
    ):

        self.layers_info = layers_info
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_num = model_num
        self.weights_path = weights_folder + "model_" + str(self.model_num) + ".h5"

        # Build the model
        self.model = self.build_model()

    def build_model(self):

        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.input_shape))

        for i in range(len(self.layers_info)):
            num_filters = self.layers_info[i][0]
            filter_size = self.layers_info[i][1]
            stride = self.layers_info[i][2]
            batch_norm = self.layers_info[i][3]
            activation = self.layers_info[i][4]
            dropout = self.layers_info[i][5]
            pool_size = self.layers_info[i][6]
            pool_type = self.layers_info[i][7]

            model.add(tf.keras.layers.Conv2D(num_filters, filter_size, strides=stride))

            if batch_norm == True:
                model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Activation(activation))

            if dropout > 0:
                model.add(tf.keras.layers.Dropout(dropout))

            if pool_type == "max":
                model.add(tf.keras.layers.MaxPool2D(pool_size))
            elif pool_type == "avg":
                model.add(tf.keras.layers.AveragePooling2D(pool_size))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.weights_path, monitor="val_loss", save_best_only=True
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30
        )

        self.callbacks = [model_checkpoint, early_stopping]

        return model

    def get_model_summary(self):
        print(self.model.summary())

    def fit(self, x_train, y_train, x_val, y_val, batch_size, epochs):

        self.hist = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
        )

    def plot_curves(self, export=False):
        os.makedirs("./plots/", exist_ok=True)
        plt.figure()
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.plot(self.hist.history["loss"], label="Training Loss")
        plt.plot(self.hist.history["val_loss"], label="Test Loss")
        plt.legend()
        loss_fig = plt.gcf()
        # plt.show()
        if export is True:
            loss_plot_path = "./plots/" + str(self.model_num) + "_loss.png"
            loss_fig.savefig(loss_plot_path, dpi=600)

        plt.figure()
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(self.hist.history["accuracy"], label="Training Accuracy")
        plt.plot(self.hist.history["val_accuracy"], label="Test Accuracy")
        plt.legend()
        acc_fig = plt.gcf()
        # plt.show()
        if export is True:
            acc_plot_path = "./plots/" + str(self.model_num) + "_acc.png"
            acc_fig.savefig(acc_plot_path, dpi=600)

    def get_metrics(self):
        self.model.load_weights(self.weights_path)
        ytestprobs = self.model.predict(X_test)
        ytrainprobs = self.model.predict(X_train)
        ypredtest = np.argmax(ytestprobs, axis=1)
        ypredtrain = np.argmax(ytrainprobs, axis=1)
        self.test_acc = accuracy_score(y_test, ypredtest)
        self.test_logloss = log_loss(y_test, ytestprobs)
        self.train_acc = accuracy_score(y_train, ypredtrain)
        self.train_logloss = log_loss(y_train, ytrainprobs)
        print("Train Accuracy = %.3f" % (self.train_acc))
        print("Test Accuracy  = %.3f" % (self.test_acc))
        print("Train Log Loss = %.3f" % (self.train_logloss))
        print("Test Log Loss  = %.3f" % (self.test_logloss))
        plt.figure()
        sns.heatmap(confusion_matrix(y_test, ypredtest), annot=True)
        heatmap_path = "./plots/" + str(self.model_num) + "_heatmap.png"
        heatmap_plot = plt.gcf()
        heatmap_plot.savefig(heatmap_path, dpi=600)
        print(classification_report(y_test, ypredtest, digits=3))
        report = classification_report(y_test, ypredtest, digits=3, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv("./plots/" + str(self.model_num) + "_report.csv")


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


############################################################
# Hyperparameters (ADJUST THIS SECTION)

"""
    Each 7-tuple of the list layers_info is a tuple with the following parameters:
    (num_filters, filter_size, stride, batch_norm, activation, dropout, pool_size, pool_type)
"""

layers_info = [
    (64, (3, 3), (1, 1), True, "relu", 0.2, 2, "max"),
    (32, (2, 2), (1, 1), True, "relu", 0.2, 2, "max"),
]

batch_size = 128
epochs = 100
model_num = 1

############################################################

cnn = CNN_Model(
    layers_info=layers_info,
    model_num=model_num,
    input_shape=X_train.shape[1:],
    num_classes=pd.Series(y_train).nunique(),
)

cnn.get_model_summary()

cnn.fit(X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=epochs)

cnn.plot_curves(export=True)
cnn.get_metrics()
