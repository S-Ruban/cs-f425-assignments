# Deep Learning Assignment 1

## CS F425 Deep Learning Assignment 1: Comparision of various ANN Architectures

### Requirements
In a terminal, open a new virtual environment, navigate to the directory containing the code (.py files) and install the required libaries by typing the following command

```
pip install requirements.txt
```

### Running All Models

To run all the models, run the bash script by typing the following command
```
bash run.sh
```

### Main hyperparameters

- Number of layers
- Number of neurons in a layer
- Activation function of neurons

### Other hyperparameters
- Batch size 
- Epochs (Set arbitrarily large because early stopping used)


### Running Individual Models
For running individual models, run the following command, which gives info on how to train a model with a certain set of hyperparameters

```
python ANN_model_scriptable.py --help
```

For example, to train a model with 1 layer, 128 neurons, ReLU activation, 32 as batch size and 1000 epochs, run the below command:
```
python ANN_model_scriptable.py -nh 1 -nn 128 -ac relu -bs 32 -ep 1000
```

### Results
After training, the console prints the accuracy and categorical cross entropy loss (log loss) of both training and test sets. Weights and plots are stored in separate folders (weights/ and plots/)

