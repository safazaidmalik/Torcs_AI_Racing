"""For Automatic Driving"""
# Number of sensors in observations
import csv
from _csv import reader
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn import preprocessing

import Trainer

epoch_count = 100

# Batch size
batch_size = 32
sensor_count = 29
action_count = 3

observations_all = np.zeros((0, sensor_count))
actions_all = np.zeros((0, action_count))
observations_list = []
actions_list = []


with open('CSVFILE.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        #for x in
        row = [float(x) for x in row]
        observations_list.append(row[0:29])
        actions_list.append(row[29:32])
        print(row)
        print(len(row))
print(observations_list[0])
print(actions_list[0])

for observation, action_made in zip(observations_list, actions_list):
    # Concatenate all observations into array of arrays
    observations_all = np.concatenate([observations_all, np.reshape(
        observation, (1, sensor_count))], axis=0)

    # Concatenate all actions into array of arrays
    actions_all = np.concatenate([actions_all, np.reshape(
        action_made, (1, action_count))], axis=0)


model = Trainer.Agent(name='model', input_num=observations_all[0].size,
                    output_num=actions_all[0].size)

# Train the model with the observations and actions availiable
model.train(observations_all, actions_all, n_epoch=epoch_count,
            batch=batch_size)



