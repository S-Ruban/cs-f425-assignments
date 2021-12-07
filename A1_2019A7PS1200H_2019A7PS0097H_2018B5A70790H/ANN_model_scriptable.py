import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, log_loss
import argparse

class ANN_Model:

    def __init__(self, 
                 num_hidden, 
                 num_neurons, 
                 activation, 
                 input_shape, 
                 num_classes, 
                 weights_folder = './weights/'):
        
        self.num_hidden = num_hidden
        self.num_neurons = num_neurons
        self.activation = activation
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = str(num_hidden)+'l_'+str(num_neurons)+'n_'+str(activation)
        self.weights_path = weights_folder+ self.name +'.h5'
    
        # Build the model
        self.model = self.build_model()


    def build_model(self):
        
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.input_shape))
        
        for i in range(self.num_hidden):
            model.add(tf.keras.layers.Dense(self.num_neurons, self.activation))
            
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
             
        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath= self.weights_path, 
            monitor='val_loss', 
            save_best_only=True
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        self.callbacks = [model_checkpoint, early_stopping]

        return model
    
    def get_model_summary(self):
        print(self.model.summary())

    def fit(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        print("batch size = "+str(batch_size)+"\n")
        self.hist = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size = batch_size,
            epochs = epochs,
            verbose = 0,
            callbacks = self.callbacks
        )
    
    def plot_curves(self, export = False):
        os.makedirs('./plots/', exist_ok=True)
        plt.figure()
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.hist.history['loss'], label = 'Training Loss')
        plt.plot(self.hist.history['val_loss'], label = 'Test Loss')
        plt.legend()
        fig = plt.gcf()
        # plt.show()
        if export is True:
            plot_path = "./plots/" + self.name + ".png"
            fig.savefig(plot_path, dpi = 600)
    
    def get_metrics(self, X_train, y_train, X_test, y_test):
        self.model.load_weights(self.weights_path)
        ytestprobs = self.model.predict(X_test)
        ytrainprobs = self.model.predict(X_train)
        ypredtest = np.argmax(ytestprobs, axis = 1)
        ypredtrain = np.argmax(ytrainprobs, axis = 1)
        self.test_acc = accuracy_score(y_test, ypredtest)
        self.test_logloss = log_loss(y_test, ytestprobs)
        self.train_acc = accuracy_score(y_train, ypredtrain)
        self.train_logloss = log_loss(y_train, ytrainprobs)
        print('Metrics for %d layers, %d neurons, %s activation: '%(self.num_hidden, self.num_neurons, self.activation))
        print('Train Accuracy = %.3f'%(self.train_acc))
        print('Test Accuracy  = %.3f'%(self.test_acc))
        print('Train Log Loss = %.3f'%(self.train_logloss))
        print('Test Log Loss  = %.3f'%(self.test_logloss))
        print()
        


def main():
    parser = argparse.ArgumentParser(description='Comparision of various ANN Architectures on Fashion MNIST dataset')
    parser.add_argument('-nh', '--num_hidden', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-nn', '--num_neurons', type=int, default=28, help='Number of neurons per hidden layer')
    parser.add_argument('-ac', '--activation', type=str, default='relu', help='Activation function', choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('-ep', '--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('-ex', '--export', type=bool, default=True, help='Export plots')
    args = parser.parse_args()
    n_hidden = args.num_hidden
    n_neurons = args.num_neurons
    activation = args.activation
    batch_size = args.batch_size
    epochs = args.epochs
    export = args.export

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])


    ann = ANN_Model(
        num_hidden=n_hidden,
        num_neurons=n_neurons,
        activation=activation,
        input_shape=X_train.shape[1:],
        num_classes=pd.Series(y_train).nunique()
    )

    ann.fit(X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=epochs)

    ann.plot_curves(export)

    ann.get_metrics(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()