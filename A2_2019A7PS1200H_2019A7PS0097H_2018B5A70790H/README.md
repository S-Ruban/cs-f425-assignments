# Deep Learning Assignment 2

## CS F425 Deep Learning Assignment 2: Convolutional Neural Networks

Edit `layers_info` to specify desired number of layers and set of hyperparameters in each layer before running.

To run the code, type in

```
python CNN_model.py
```

### Main hyperparameters

- Number of convolutional filters
- Filter size
- Stride length
- Activation function
- Dropout
- Pool size
- Pool type

### Other hyperparameters

- Batch size
- Epochs (Set arbitrarily large because early stopping used)
- Number of neurons in a layer

### Results

After training, the console prints the accuracy and categorical cross entropy loss (log loss) of both training and test sets. Weights and plots are stored in separate folders (`/weights/ `and `/plots/`)
