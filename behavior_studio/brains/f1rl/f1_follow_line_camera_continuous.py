#!/usr/bin/env python

'''
Based on:
=======

'''
import json
import os
import time
from distutils.dir_util import copy_tree

import gym
import gym_gazebo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym import logger, wrappers
from keras import backend as K


# To equal the inputs, we set the channels first and the image next.
K.set_image_data_format('channels_first')


def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)


def plot_durations():
    plt.figure(1)
    plt.clf()
    
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)
    
    # Take 100 episode averages and plot them too
    if step % 10 == 0:
        mean_episode = np.mean(episode_durations)
        plt.plot(mean_episode)

    plt.pause(0.001)  # pause a bit so that plots are updated



episode_durations = []



####################################################################################################################
# MAIN PROGRAM
####################################################################################################################
if __name__ == '__main__':

    #REMEMBER!: turtlebot_cnn_setup.bash must be executed.
    env = gym.make('GazeboF1CameraEnvContinuous-v0')
    observation, pos = env.reset()
    
    while True:

        action = env.action_space.sample() 
        newObservation, reward, done, _ = env.step(action)
        print(newObservation)
        env._flush(force=True)
        if done:
            # break
            env.reset()
        
    env.close()
