import csv
from _csv import writer, reader
from csv import DictWriter

import numpy as np
import pygame

import ManualMovement
import msgParser
import carState
import carControl
import telemetry
import Trainer

epoch_count = 100
manualkeys = False
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


class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage, interface, expert):
        '''Constructor'''
        self.observations_all = None
        self.obs = None
        self.act = ManualMovement.Action()
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.interface = interface
        self.expert = expert
        self.action_list = []
        self.observation_list = []
        self.parser = msgParser.MsgParser()

        self.state = carState.CarState()

        self.control = carControl.CarControl()

        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None

    def init(self):
        '''Return init string with rangefinder angles'''

        self.angles = [0 for x in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):

        # telemetry.getTelemetry(self.state)

        # self.interface.get_key_state()


        if manualkeys == True:

            self.act = self.expert.get_action(self.act)


            #print(f"Expert Act = {self.act.accel}")

            # Normalizing

            self.state.setFromMsg(msg)

            self.state.normalize_obs()
            obs_list = self.state.get_obs(angle=True, gear=True, rpm=True,
                                   speedX=True, speedY=True, track=True,
                                   trackPos=True, wheelSpinVel=True)

            self.observation_list.append(obs_list)

            self.act.normalize_act()
            act_list = self.act.get_act(gas=True, gear=True, steer=True)
            self.action_list.append(act_list)
            self.act.un_normalize_act()

            self.step(self.act)
            #self.act.un_normalize_act()
            # self.gear()
            #self.step(self.act)

            return self.control.toMsg()
        else:
            self.state.setFromMsg(msg)

            self.state.normalize_obs()
            obs_list = self.state.get_obs(angle=True, gear=True, rpm=True,
                                          speedX=True, speedY=True, track=True,
                                          trackPos=True, wheelSpinVel=True)

            self.observation_list.append(obs_list)


            # Normalize the act and add it to list of actions
            # Important to un-normalize the act before sending it to torcs
            act_list = model.predict(np.reshape(obs_list, (1, sensor_count)))
            self.act.set_act(act_list, gas=True, gear=True, steer=True)
            self.act.un_normalize_act()

            # Execute the action and get the new observation
            self.step(self.act)
            return self.control.toMsg()

    def step(self, act):
        self.control.setAccel(act.accel)
        self.control.setBrake(act.brake)
        self.control.setClutch(act.clutch)
        self.control.setGear(act.gear)
        self.control.setSteer(act.steer)
        self.control.setMeta(act.meta)
        self.control.focus = act.focus

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos

        # self.control.setSteer((angle - dist*0.5)/self.steer_lock)

    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()

        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False

        if up and rpm > 7000:
            gear += 1

        if not up and rpm < 3000:
            gear -= 1

        self.control.setGear(gear)

    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()

        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0

        self.control.setAccel(accel)

    def onShutDown(self):

        #print(self.observation_list)
        #print(self.action_list)
        field_names = ["angle", "curLapTime", "damage",
                   "distFromStart", "distRaced", "fuel",
                   "gear", "lastLapTime", "opponents", "racePos",
                   "rpm", "speedX", "speedY", "speedZ", "track",
                   "trackPos", "wheelSpinVel", "z", "focus", "x",
                   "y", "roll", "pitch", "yaw", "speedGlobalX",
                   "speedGlobalY", "accel", "brake", "gas", "clutch", "gear",
                   "steer", "focus", "meta"]
        with open('CSVFILE.csv', 'a+', newline='') as f_object:
            # Pass the file object and a list
            # of column names to DictWriter()
            # You will get a object of DictWriter
            # dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            csv_writer = writer(f_object)
            # Pass the dictionary as an argument to the Writerow()

            for observation, action_made in zip(self.observation_list, self.action_list):
                list_of_elem = []
                list_of_elem.extend(observation)
                list_of_elem.extend(action_made)
                csv_writer.writerow(list_of_elem)




    def onRestart(self):
        pass
