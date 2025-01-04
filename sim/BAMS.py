# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 04:48:06 2025

@author: Murad
"""
import numpy as np
import matplotlib.pyplot as plt
from sim.Simulator import Simulator
from collections import deque
import copy
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
import os
from datetime import datetime
import pandas as pd
from utils.misc_functions import plot_custom, make_video, Custom_marker, save_parameters, plot_custom2
import torch
import random

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ModelDynamics:
    '''
    Input x(position) and measured z(power) have non-linear relationship.
    We can describe their relation with two functions.
        1. x_t = f(x_(t-1), u_(t)): stateTransitionFunction
        2. z_t = h(x_t): MeasurementFunction
    These functions will be used in Unscented kalman filter.
    '''
    def __init__(self, simulator = []):
        self.sim = simulator
    
    def stateTransitionFunction(self, x, dt, del_x): 
        '''
        Parameters
        ----------
        x : TYPE: numpy array: shape [n,]
            DESCRIPTION: position in previous time step
        del_x : TYPE: numpy array: shape [ n,]
            DESCRIPTION: movement for small time

        Returns
        -------
        x_2 : TYPE: numpy array: shape [ n,]
            DESCRIPTION: x_2 = x + del_x
        '''
        x_2 = x + del_x
        return x_2
    
    def measurementFunction(self, x):
        """ measurement function - convert state into a measurement """
        z = self.sim.get_cumulative_power_for_interrogators(x)
        return np.asarray(z).astype(np.float32)
    
    def reset(self, sim):
        self.sim = sim


class Estimate_velocity:
    def __init__(self):
        self.window = 100
        self.deque_velocity = deque([], maxlen = self.window)
        
    def push_displacement(self, displacement, dt):
        self.deque_velocity.append(displacement/dt)
        
    def estimate_velocity(self):
        if len(self.deque_velocity) < self.window:
            estimated_value = self.deque_velocity[0] * 0 # zero velocity
        else:
            estimated_value = np.mean(np.asarray(self.deque_velocity)[-self.window:], axis = 0)
        return estimated_value
    
    def reset(self):
        self.deque_velocity = deque([], maxlen = self.window)
        


        
        
class BAMS:
    def __init__(self, args):
        """ simulator parameters """
        self.args = args
        self.initial_nodes_position = np.array(self.args.initial_nodes_position)
        self.interrogators_position = np.array(self.args.interrogators_position).reshape(-1, 6)
        self.x_cor_min, self.x_cor_max = self.args.x_boundary[0], self.args.x_boundary[1] 
        self.y_cor_min, self.y_cor_max = self.args.y_boundary[0], self.args.y_boundary[1]
        self.slope_angle = self.args.slope_angle
        self.dt = self.args.dt
        self.velocity_model_id = self.args.velocity_model_id
        self.sim = Simulator(slope_angle = self.slope_angle, 
                             x_cor = [self.x_cor_min, self.x_cor_max], 
                             y_cor = [self.y_cor_min, self.y_cor_max], 
                             initial_nodes_position = self.initial_nodes_position, 
                             interrogators_position = self.interrogators_position.tolist(),
                             velocity_model_id = self.velocity_model_id)
        self.sim.delta_T = self.dt 
        self.trues, self.estimates = [], []
        
        """ UKF parameters """
        self.ukf_alpha = self.args.ukf_alpha # alpha: (0, 1]
        self.ukf_beta = self.args.ukf_beta # beta: 2.0
        self.ukf_kappa = self.args.ukf_kappa # kappa: >= 0 # lambda = alpha^2 * (n + kappa) - n
        self.ukf_R_factor = self.args.ukf_R_factor # measurement noise
        self.ukf_Q_factor = self.args.ukf_Q_factor # process noise
        self.ukf_P_factor = self.args.ukf_P_factor
        self.path = self.args.path
        
        self.number_of_interrogators = len(self.sim.interrogator)
        self.number_of_nodes = int(self.initial_nodes_position.shape[0] / 2)
        self.x_mesh, self.y_mesh = np.meshgrid(np.linspace(self.x_cor_min, self.x_cor_max, 10), np.linspace(self.y_cor_min, self.y_cor_max, 10))
        self.z_mesh = self.sim.slope_plane(self.x_mesh.reshape(-1), self.y_mesh.reshape(-1)).reshape((self.x_mesh.shape[0], self.x_mesh.shape[1]))
        self.starting_positions_of_nodes = self.sim.initial_positions_of_nodes # [-2, 5, 0, 5, +2, 5] # x_n1, y_n1, x_n2, y_n2, x_n3, y_n3
        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    
    def run(self):
        """ Running the function for unscented kalman filter method"""
        # Directories
        directory = self.path + 'ukf_' + self.current_time + '/'
        dir_frames = directory + 'frames/'
        dir_video = directory + 'video/'
        dir_other = directory + 'other/'
        os.makedirs(dir_frames, exist_ok = True)
        os.makedirs(dir_video, exist_ok = True)
        os.makedirs(dir_other, exist_ok = True)
        
        self.sim.reset()
        dataset = './data/ukf/data.csv'
        mdynamics= ModelDynamics(simulator = self.sim)
        Es_velocity = Estimate_velocity()
        
        mdynamics.reset(self.sim)
        Es_velocity.reset()
        

            
        ################################################################
        """ Unscented Kalman Filter """
        points = MerweScaledSigmaPoints(self.number_of_nodes * 2,
                                        alpha = self.ukf_alpha, 
                                        beta = self.ukf_beta, 
                                        kappa = self.ukf_kappa)
        ukf = UnscentedKalmanFilter(dim_x = self.number_of_nodes * 2, 
                                    dim_z = self.number_of_interrogators,
                                    dt = self.dt, 
                                    fx = mdynamics.stateTransitionFunction, 
                                    hx = mdynamics.measurementFunction, 
                                    points = points)
        
        ukf.x = self.starting_positions_of_nodes.astype(np.float32)
        ukf.R = np.eye(self.number_of_interrogators) * self.ukf_R_factor # measurement noise
        
        process_vari = np.ones_like(self.initial_nodes_position, dtype = np.float32)
        process_vari[0::2] *= self.ukf_Q_factor / 10 # process noise would be less in x direction
        process_vari[1::2] *= self.ukf_Q_factor # process noise would be high in y direction compared to x direction
        ukf.Q = np.eye(ukf.x.shape[0]) * process_vari # process noise
        ukf.P *= self.ukf_P_factor
        ukf_P_list = []
        ukf_P_list.append(ukf.P.diagonal())
        old_ukf_x = copy.deepcopy(ukf.x)
        self.trues, self.estimates = [], [] # position of the nodes
        
        ########## getting data from the dataset
        data = pd.read_csv(dataset).values # [xy_pos_nodes, int_power, xy_vel, xy_disp]
        ########################################
        true_velocity, estimated_velocity = [], []
        i = 0
        
        while True:
            true_positions = data[i, :self.number_of_nodes * 2]
            z = data[i, self.number_of_nodes * 2: self.number_of_nodes * 2 + self.number_of_interrogators]
            z = np.asarray(z).reshape(-1, 1).astype(np.float32)
            tr_velocity = data[i, -4*self.number_of_nodes: -2*self.number_of_nodes]
            displacement = data[i, -2*self.number_of_nodes: ]
        
            """ Unscented kalman filter """
            estimated_displacement = ukf.x - old_ukf_x
            estimated_displacement[0::2] = estimated_displacement[0::2] * 0 # as x disp is very small and random, we are assuming zero displacement 
            estimated_displacement[1::2] = estimated_displacement[1::2] * (estimated_displacement[1::2] <0) # y displacement can not be positive.
        
            # # Assumption-01: displacement is same as previous time steps
            estimated_displacement = estimated_displacement *0
            # Assumption-02: node's y position can not be greater than 10 , as they are moving downward.
            ukf.x[1::2][ukf.x[1::2]>10] = 10
            # Assumption-03: node's y position can not be less than -10 , we are considering it inside the frame
            ukf.x[1::2][ukf.x[1::2]<-10] = -10
            
            Es_velocity.push_displacement(estimated_displacement, self.dt)
            old_ukf_x = copy.deepcopy(ukf.x)
            est_velocity = Es_velocity.estimate_velocity() # row vector

            try:
                ukf.predict(del_x = est_velocity * self.dt) # del_x misc.fx arguments
            except:
                print('check for error')
                return None
            
            ukf.update(z.reshape(-1)) # z: row vector
            ukf_P_list.append(ukf.P.diagonal())
            self.estimates.append(ukf.x)
            self.trues.append(true_positions)
            
            ######################################### plotting frames #############################################
            if i == 0: # initialize all the directory and figure at first run                
                # %matplotlib qt
                CM = Custom_marker()
                # fig = plt.figure(figsize=(8, 6), dpi=100) # for high resolution only\
                fig = plt.figure()
                ax = fig.add_subplot(projection = '3d')
                ax.view_init(elev=10, azim=-65)
            else:   
                # plotting the 3D figure and saving the frames
                plt.cla()
                plt.title("alpha: {}, beta: {}, kappa: {}, P: {}, Q: {}, R: {}".format(self.ukf_alpha, self.ukf_beta, self.ukf_kappa, self.ukf_P_factor, self.ukf_Q_factor, self.ukf_R_factor))
                ax.xaxis.label.set_size(10)
                ax.yaxis.label.set_size(10)
                ax.zaxis.label.set_size(10)
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                ax.tick_params(axis='z', labelsize=6)
                ax.plot_surface(self.x_mesh, self.y_mesh, self.z_mesh, color = '#8B4513', alpha=0.2)
                ax.scatter(self.interrogators_position[:,0], self.interrogators_position[:,1], self.interrogators_position[:,2], color = 'blue', marker = CM.interrogator, s = 200, label = 'Interrogator')
                # ax.scatter(ukf.x[0::2], ukf.x[1::2], self.sim.slope_plane(ukf.x[0::2], ukf.x[1::2]), color = 'green', marker = CM.node, s = 100, label = 'estimated_ukf')
                ax.scatter(true_positions[0::2], true_positions[1::2], self.sim.slope_plane(true_positions[0::2], true_positions[1::2]), color = 'r', marker = CM.node, s = 100, label = 'original')
                plt.rcParams.update({'font.size': 5})
                ax.set_xlim(self.x_cor_min, self.x_cor_max)
                ax.set_ylim(self.y_cor_min, self.y_cor_max)
                ax.set_zlim(self.x_cor_min, self.x_cor_max)
                ax.set_xlabel('x axis')
                ax.set_ylabel('y axis')
                ax.set_zlabel('z axis')
                ##########################
                # ax.xaxis.pane.fill = False
                # ax.yaxis.pane.fill = False
                # ax.zaxis.pane.fill = False
                # ax.grid(False)
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.set_zticks([])
                # # Set thick axis lines
                # ax.w_xaxis.line.set_lw(0.5)  # X-axis line width
                # ax.w_yaxis.line.set_lw(0.5)  # Y-axis line width
                # ax.w_zaxis.line.set_lw(0.5)  # Z-axis line widt
                ################################
                plt.rcParams.update({'font.size': 8})
                plt.legend(fontsize='small')
                
                plt.savefig(dir_frames + '{}_fig.png'.format(i), dpi = 500)
                plt.draw()
                plt.pause(0.001)
            ######################################################################################
            
            i += 1
            if i % 10 == 0:
                completed = (self.y_cor_max - ukf.x[1::2].min()) / (self.y_cor_max - self.y_cor_min) * 100
                print('{}: {:.1f}% completed'.format(i, completed))
            if np.any(true_positions[1::2] <= self.y_cor_min): # break the loop, if any of the node's y position passes the minimum y coordinate of the frame.
                break
            
            true_velocity.append(tr_velocity)
            estimated_velocity.append(Es_velocity.estimate_velocity())
            
        """ Unscented Kalman Filter """
        self.estimates = np.asarray(self.estimates)
        self.trues = np.asarray(self.trues)

        ###########
        plot_custom(self.number_of_nodes, np.asarray(estimated_velocity), -np.asarray(true_velocity), x_label = 'steps', y_label = 'velocity', path = dir_other, name = self.current_time, title = self.args)
        plot_custom(self.number_of_nodes, self.estimates, self.trues, x_label = 'steps', y_label = 'x-y positions', path = dir_other, name = self.current_time, title = self.args)
        make_video(input_path = dir_frames, output_path = dir_video, name = self.current_time + '_video.mp4')
        mse = np.mean((self.trues - self.estimates)**2)
        save_parameters(self.args, dir_other, self.current_time +'.txt', self.estimates, self.trues, loss = mse)
        # plt.close('all')
        return mse
    
    def generate_datasets(self):
        self.sim.reset()
        pos, power, vel, dis = [], [], [], []
        while True:
            nodes_position_t2, power_t2, velocity_t1, displacement_t1 = self.sim.get_data()  
            """
            nodes_position_t2: position of the nodes at time t
            power_t2: cumulative received power of the interrogators at time t
            velocity_t1: velocity of the nodes at time (t-1)
            displacement_t1 : nodes_position_t2 = nodes_position_t1 + displacement_t1 = nodes_position_t1 + velocity_t1*dt

            """        
            pos.append(nodes_position_t2.tolist())
            power.append(power_t2)
            vel.append(velocity_t1)
            dis.append(displacement_t1)
            if np.any(nodes_position_t2[1::2] <= self.y_cor_min): # break the loop, if any of the node's y position passes the minimum y coordinate of the frame.
                break
        pos = np.asarray(pos)
        power = np.asarray(power)
        vel = np.asarray(vel)
        dis = np.asarray(dis)
        
        data = np.concatenate((pos, power, vel, dis), axis = 1)
        df = pd.DataFrame(data)
        #####################
        # vel0 = vel[255:, :]
        # power0 = power[255: , :]
        # nrows, ncols = 4, 1
        # fig, axs = plt.subplots(nrows, ncols, figsize=(6, 10))
        # axs = axs.reshape(nrows, ncols)
        # n, num_figures = 0, 4
        # i, j = 0, 0
        # axs[i, j].plot(vel0[:, 0])
        # axs[i, j].set_title('{}: Node-{}'.format('velocity in x direction', 1))
        # axs[i, j].set_xlabel('time steps')
        # axs[i, j].set_ylabel('Vx')
        # axs[i, j].legend()
        # plt.rcParams.update({'font.size': 8})
        # plt.tight_layout()
        
        # i, j = 1, 0
        # axs[i, j].plot(vel0[:, 1])
        # axs[i, j].set_title('{}: Node-{}'.format('velocity in y direction', 1))
        # axs[i, j].set_xlabel('time steps')
        # axs[i, j].set_ylabel('Vy')
        # axs[i, j].legend()
        # plt.rcParams.update({'font.size': 8})
        # plt.tight_layout()

        # i, j = 2, 0
        # axs[i, j].plot(power0[:, 0])
        # axs[i, j].set_title('{}: Interrogator-{}'.format('Cumulative power signal', 1))
        # axs[i, j].set_xlabel('time steps')
        # axs[i, j].set_ylabel('dBm')
        # axs[i, j].legend()
        # plt.rcParams.update({'font.size': 8})
        # plt.tight_layout()
        
        # i, j = 3, 0
        # axs[i, j].plot(power0[:, 1])
        # axs[i, j].set_title('{}: Interrogator-{}'.format('Cumulative power signal', 2))
        # axs[i, j].set_xlabel('time steps')
        # axs[i, j].set_ylabel('dBm')
        # axs[i, j].legend()
        # plt.rcParams.update({'font.size': 8})
        # plt.tight_layout()  
        # plt.savefig('velocity.png', dpi = 500)
        ###########
        
        # plot
        dir_other = self.path + 'gen_data_{}_seed{}_'.format(self.args.velocity_model_id, self.args.seed) + self.current_time + '/'
        os.makedirs(dir_other, exist_ok = True)
        plot_custom2(vel[:, 0::2], self.number_of_nodes, title = 'x-velocity', sub_title = 'node', x_label = 'steps',
                     y_label = 'vel', path = dir_other, name = self.current_time)
        plot_custom2(vel[:, 1::2], self.number_of_nodes, title = 'y-velocity', sub_title = 'node', x_label = 'steps',
                     y_label = 'vel', path = dir_other, name = self.current_time)
        plot_custom2(power, self.number_of_interrogators, title = 'power', sub_title = 'interrogator', x_label = 'steps',
                     y_label = 'power', path = dir_other, name = self.current_time)
        # dataset
        self.date_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(dir_other + '\{}_seed{}_'.format(self.args.velocity_model_id, self.args.seed) + self.date_stamp + '.csv', index = False)
        # dataset info file
        with open(dir_other + '\{}_seed{}_'.format(self.args.velocity_model_id, self.args.seed) + self.date_stamp + '.txt', 'w') as f:    
            for key, value in vars(self.args).items():
                f.write(f'{key}: {value}\n') 
        return 
    

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)