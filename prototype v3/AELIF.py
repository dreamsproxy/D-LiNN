import matplotlib.pyplot as plt
import numpy as np
import random
import math
from MyPlot import my_plot
e = math.e
# A class for Adaptive-ELIF Model.
class AELIF:
    # this function will initialize model.
    # no_fig will be used in my_plot function.
    def __init__(self, no_fig, dt, u_rest=-70, R=10, I=0, tau_m=8, thresh=-50, delta=2, a=0.5, b=0.5, tau_w=100,
                 duration=20):
        self.fig = no_fig
        self.dt = 1
        self.potential_rest = u_rest
        self.R = R
        self.Current = I
        self.tau_m = tau_m
        self.thresh = thresh
        self.delta = delta
        self.a = a
        self.b = b
        self.tau_w = tau_w
        self.duration = duration
        self.potential_spike = -40
        self.w = []
        self.spike = []
        self.time = []
        self.current_lst = []
        self.potential = []
        
        #   Pregenerate time (ticks in ms)
        #   Pregenerate potential log
        #   Pregenerate discrete log
        for i in range(0, int(duration/dt), 1):
            print(i*dt)
            
            self.time.append(i * dt)
            self.potential.append(0)
            self.w.append(0)
        #print(self.potential)
        #print(self.w)
        self.current()
        self.calc_potential()
        #print(self.potential)
        #print(self.w)
        return

    # this function will be used for making a list of currents
    # if self.Current is -1 , it means that the user wants random currents in all times.
    # otherwise the currents will be fixed.
    def current(self):
        if self.Current != -1:
            for i in range(len(self.time)):
                if i < len(self.time) // 10:
                    print(i)
                    self.current_lst.append(0)
                else:
                    self.current_lst.append(self.Current)
        else:
            for i in range(len(self.time)):
                if i < len(self.time) // 10:
                    self.current_lst.append(0)
                else:
                    self.current_lst.append(random.randrange(-20, 100, 1) / 10)
        return

    # this function will calculate w which will be used in calculation of potential.
    # consider that values of w or any other continuous parameter will be stored in list
    # so it will become discrete.
    
    #   Allows the recaclulation using the previous spike instead of instantly
    #   calculating the updated data.
    #       I DONT KNOW WHY
    #       LETS FUCK AROUND AND FIND OUT
    
    def calc_w(self, i):
        t_fire = -1
        if len(self.spike) >= 1:
            t_fire = self.spike[-1]
        diff = self.a * (self.potential[i - 1] - self.potential_rest) - self.w[i - 1] + self.b * self.tau_w * int(1 - np.sign(self.time[i - 1] - t_fire))
        tmp = diff / self.tau_w * self.dt
        self.w[i] = self.w[i-1] + tmp
        return

    # this function calculates potential and stores them in a list.
    # each time that neuron spikes, its time will be stored in a list named spike list.
    def calc_potential(self):
        self.potential[0] = self.potential_rest
        self.w[0] = 0
        for i in range(1, len(self.time)):
            #print(f"tick {i}")
            self.calc_w(i)
            diff = -1 * (self.potential[i - 1] - self.potential_rest) + np.exp((self.potential[i - 1] - self.thresh) / self.delta) * self.delta\
                   + self.R * self.current_lst[i] - self.R * self.w[i]
            tmp = diff / self.tau_m * self.dt + self.potential[i - 1]
            #   If spike
            if tmp >= self.thresh:
                self.potential[i-1] = self.potential_spike
                self.potential[i] = self.potential_rest
                self.spike.append(self.time[i])
                #print("\nSpiked!")
                #print(self.potential[i])
            else:
                self.potential[i] = tmp
                #print(self.potential[i])
        return