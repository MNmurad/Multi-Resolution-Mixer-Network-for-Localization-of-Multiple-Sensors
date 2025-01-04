# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:35:25 2024

@author: Murad
"""
# import matlab.engine
import torch
import numpy as np
import copy
from sim.SimulatorMatPY import SimulatorMatPY


class Simulator:
    def __init__(self, slope_angle = [],
                 x_cor = [],
                 y_cor = [],
                 initial_nodes_position = [],
                 interrogators_position = [], 
                 velocity_model_id = []):
        '''
        Parameters
        ----------
        slope_angle : degree of the slop of the plane
        x_cor : [Plane's minimum x coordinate, plane's maximum x coordinate]
        y_cor : [Plane's minimum y coordinate, plane's maximum y coordinate]
        initial_nodes_position : np.array[node1.x, node1.y, node2.x, node2.y, ...............]
        interrogators_position : [[int-1_x, int-1_y, int-1_z, int-1_theta, int-1_phi, 1], ......[int-i_x, int-i_y, int-i_z, int-i_theta, int-i_phi, 1]]
        '''
        self.simMat = SimulatorMatPY()
        self.slope_angle_rad = torch.tensor(slope_angle * 3.14159/ 180)  # converting degree to radian
        self.int_locations = interrogators_position # [[-5, 5, 50, 1.571, 1.571, 1], [5, 5, 50, 1.571, 1.571, 1]]
        self.node_theta1, self.node_theta2, self.node_r = 1.571, 1.571, 1
        self.initial_positions_of_nodes = initial_nodes_position # np.array([-2, 5, 0, 5, +2, 5]) # x_n1, y_n1, x_n2, y_n2, x_n3, y_n3
        self.current_positions_of_nodes = copy.deepcopy(self.initial_positions_of_nodes)
        
        self.interrogator = {}
        for i in range(0, len(self.int_locations)):
            self.interrogator[i] = self.int_creation(self.int_locations[i][0], self.int_locations[i][1],
                                             self.int_locations[i][2], self.int_locations[i][3],
                                             self.int_locations[i][4], self.int_locations[i][5])
        self.natural_movement_toward_slope = 0.5 # value/s
        self.delta_T = 0.1
        self.time = 0 # running time for the simulator. Only updated in def model_velocity() function
        
        model_name = velocity_model_id
        # model1: used for generate training dataset (with random seed 42), random seed 43 used for generating testing set
        velocity_model = {'train': {'des': 'train',
                                     'magnitude_factor': np.random.uniform(1, 1.5, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     'start_point_factor': np.random.uniform(1, 2, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     },
                          'vali': {'des': 'vali',
                                     'magnitude_factor': np.random.uniform(2, 2.5, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     'start_point_factor': np.random.uniform(3, 3.5, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     },
                          'test-1': {'des': 'test-1',
                                     'magnitude_factor': np.random.uniform(2.5, 3, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     'start_point_factor': np.random.uniform(4, 4.5, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     },
                          'test-2': {'des': 'test-2',
                                     'magnitude_factor': np.random.uniform(1, 1.5, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     'start_point_factor': np.random.uniform(5, 5.5, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     },
                          'test-3': {'des': 'test-3',
                                     'magnitude_factor': np.random.uniform(1.5, 2, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     'start_point_factor': np.random.uniform(5.5, 6, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     },
                          'test-4': {'des': 'test-4',
                                     'magnitude_factor': np.random.uniform(1.5, 2, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     'start_point_factor': np.random.uniform(0.5, 1, size = (int(len(self.initial_positions_of_nodes)/2), )),
                                     }
                              }
        
        self.a = velocity_model[model_name]['magnitude_factor'] # velocity magnitude factors
        self.b = velocity_model[model_name]['start_point_factor'] # velocity rising starting point factors
        
        
    def node_creation(self, x, y, z, theta1, theta2, r):
        """ This function will create the node dictionary"""
        node = self.simMat.node_creation(x, y, z, theta1, theta2, r)
        return node
    
    
    def int_creation(self, x, y, z, theta1, theta2, r):
        """ This function will create the Interrogator dictionary"""
        interrogator = self.simMat.int_creation(x, y, z, theta1, theta2, r)
        return interrogator
    
    def get_data(self):
        """
        Returns
        -------
        positions: np array: shape (n, )
        power: list
        velocity : np array: shape (n, )
        displacement : np array: shape (n, )
        """
        velocity = self.model_velocity(std = 0.1, add_noise_with_velocity = True) # array([node1.Vx, node1.Vy, node2.Vx, node2.Vy..............])
        displacement =  velocity * self.delta_T
        self.current_positions_of_nodes = self.current_positions_of_nodes - displacement
        # returning positions, power, velocity, displacement
        return self.current_positions_of_nodes, self.get_cumulative_power_for_interrogators(self.current_positions_of_nodes), velocity, displacement
    
    def model_velocity(self, std = 0.1, add_noise_with_velocity = False):
        self.time = self.time + self.delta_T
        velocity_xy = np.zeros_like(self.initial_positions_of_nodes)
        velocity_xy[1::2] = self.a / (1 + (np.e) ** -(4*(self.time - self.b))) # a / (1 + e^-4(t-b))
        # # velocity of y
        # velocity_xy[1::2] = 1/ (1 + (np.e)**-(self.time ** p[1::2] - 4)) # velocity in y direction
        # # velocity of x
        # # velocity_xy[0::2] -->> zero for x velociy
        
        if add_noise_with_velocity == True:
            velocity_xy = velocity_xy + np.random.normal(loc = 0.0, scale = std, size = (len(self.initial_positions_of_nodes),))
        return velocity_xy
    
    def get_cumulative_power_for_interrogators(self, current_positions_of_nodes):
        '''
        Parameters
        ----------
        current_positions_of_nodes : TYPE: array_np, shape: [2n] # x_n1, y_n1, x_n2, y_n2, x_n3, y_n3
            DESCRIPTION.
        Returns
        -------
        power : TYPE: list: [p1, p2]
            DESCRIPTION.

        '''
        current_positions_of_nodes = current_positions_of_nodes.reshape(-1)
        nodes_x = current_positions_of_nodes[0::2]
        nodes_y = current_positions_of_nodes[1::2]
        power = []
        for i in range(0, len(self.int_locations)):
            nodes = []
            for (x, y) in zip(nodes_x, nodes_y):
                node = self.node_creation(x, y, self.slope_plane(x, y), self.node_theta1, self.node_theta2, self.node_r)
                nodes.append(node)
                
            cumulative_power, _ = self.power_calculated(nodes, self.interrogator[i])
            power.append(cumulative_power)
        return power
    
    def slope_plane(self, x, y):
        if type(y) == np.ndarray:
            y = torch.from_numpy(y)
            z = y * torch.tan(self.slope_angle_rad)
            z = z.numpy()
        else:
            z = y * torch.tan(self.slope_angle_rad)
        return z
    
    def power_calculated(self, node, interrogator):
        '''
        Parameters
        ----------
        node : TYPE: dictionary for a single node
            DESCRIPTION.
        interrogator : TYPE: dictionary for a  single interrogator
            DESCRIPTION.

        Returns
        -------
        power : TYPE: scalar, power value for the particular node and interrogator
            DESCRIPTION.

        '''
        total_power, total_voltage = self.simMat.power_calculated(node, interrogator) # node can be a single node or multiple node.
        return total_power, total_voltage
    
    def powerSingleNodeMultipleInterrogators(self, x, y, z):
        '''
        This functin will give the power, real_volt, imag_volt for a single node, which interacts with multiple interrogators.
        output shape: [power of int1, power of int2,...............]
        this function is used to generate a dataset for the whole boundary for a single node.
        x, y, z are node position. theta, phi, r are constant which are taken from this class.
        '''
        node = self.node_creation(x, y, z, self.node_theta1, self.node_theta2, self.node_r)
        power = []
        real_voltage = []
        imag_voltage = []
        for i in range(len(self.interrogator)):  
            total_power, total_voltage = self.power_calculated(node, self.interrogator[i])
            power.append(total_power)
            real_voltage.append(np.real(total_voltage))
            imag_voltage.append(np.imag(total_voltage))
        return power, real_voltage, imag_voltage
    
    def close(self):
        pass
        return 
        
    def reset(self):
        self.time = 0
        self.current_positions_of_nodes = copy.deepcopy(self.initial_positions_of_nodes)
        

