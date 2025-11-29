import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import pandas as pd
data = pd.read_csv('dataset03.csv')

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet

# Prepare dataset
ds = SupervisedDataSet(train.shape[1]-1, 1)  # last column is target y
for i, row in train.iterrows():
    x_values = row[:-1].values
    y_value = row[-1]
    ds.addSample(x_values, y_value)

# Create feedforward network
net = buildNetwork(train.shape[1]-1, 5, 1, bias=True)  # 5 hidden neurons (example)

# Train network
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(50)  # train for 50 epochs

# Save model
from pybrain.tools.xml.networkwriter import NetworkWriter
NetworkWriter.writeToFile(net, 'UE_05_App3_ANN_Model.xml')
