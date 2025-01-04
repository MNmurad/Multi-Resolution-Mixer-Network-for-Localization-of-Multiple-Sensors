# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:25:44 2024

@author: Murad
"""

import numpy as np
from scipy.io import loadmat


class SimulatorMatPY:
    '''
    Here, I converted all the matlab code to python code. 
    Because loading the conversion gain again and again takes so much time.
    This code will reduce this computational complexity.
    '''
    def __init__(self):
        self.data = loadmat('./matlab_files/Conversiongain1.mat')
        self.Conversiongain1 = self.data['Conversiongain1']
        
    def unit_vector(self, theta, phi, r):
        '''
        # matlab code
        function UV=Unit_Vector(theta,phi,r)
        x_v = r*sin(theta)*cos(phi);
        y_v = r*sin(theta)*sin(phi);
        z_v = r*cos(theta);
        E_field_node = [x_v y_v z_v];
        UV=E_field_node./norm(E_field_node);
        end
        '''
        x_v = r * np.sin(theta) * np.cos(phi)
        y_v = r * np.sin(theta) * np.sin(phi)
        z_v = r * np.cos(theta)
        
        E_field_node = np.array([x_v, y_v, z_v])
        UV = E_field_node / np.linalg.norm(E_field_node)
        
        return UV


    def position(self, x, y, z):
        '''
        # matlab code
        function int_pos = position(x,y,z)
        int_pos=[x y z];%%interrogator position
        end
        '''
        int_pos = np.array([x, y, z])  # interrogator position
        return int_pos


    def plf(self, unit_node, unit_int):
        '''
        # matlab
        function polLF_db=PLF(Unit_node,Unit_Int)
        polLF=(dot(Unit_node,Unit_Int))^2;
        polLF_db=20*log10(polLF);
        end
        '''
        polLF = np.dot(unit_node, unit_int) ** 2
        polLF_db = 20 * np.log10(polLF)
        return polLF_db

    def pathloss(self, d, f):
        '''
        # matlab
        function p=pathloss(d,f)
        c=299792458 ;
        p=20*log10((4*pi*d*f)/c);
        end
        '''
        c = 299792458  # speed of light in m/s
        p = 20 * np.log10((4 * np.pi * d * f) / c)
        return p

    def node_creation(self, x, y, z, phi, theta, r):
        '''
        # matlab
        function node_value=node_creation(x,y,z,phi,theta,r)
        nodes1.x=x;
        nodes1.y=y;
        nodes1.z=z;
        nodes1.phi=phi;
        nodes1.theta=theta;
        nodes1.r=r;
        node_value=nodes1;
        end
        '''
        nodes1 = {'x': x, 'y': y, 'z': z, 'phi': phi, 'theta': theta, 'r': r}
        return nodes1

    def int_creation(self, x, y, z, phi, theta, r):
        '''
        # matlab
        function int_value=int_creation(x,y,z,phi,theta,r)
         int1.x=x;
         int1.y=y;
         int1.z=z;
         int1.phi=phi;
         int1.theta=theta;
         int1.r=r;
         int_value=int1;
         end
        '''
        int1 = {'x': x, 'y': y, 'z': z, 'phi': phi, 'theta': theta, 'r': r}
        return int1

    def cart2sph(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        return azimuth, elevation, r


    def distance_power_sim(self, Power_transmitted, x, y, z, x_int, y_int, z_int, gain_fdrfo, int_gainfo, PLF):
        c = 299792458  # speed of light in m/s
        fo = 0.915e9  # first harmonic of signal in Hz
        second_harmonic = 2 * fo  # second harmonic of signal in Hz
        
        interrogator = np.array([x_int, y_int, z_int])  # position in meters
        node1 = np.array([x, y, z])  # position in meters
        Pt = Power_transmitted
        
        gain_fdr2fo = gain_fdrfo  # fdr gain at the moment
        gain_fdrfo = 0  # fdr gain 0 at the moment
        
        int_gain2fo = int_gainfo
        int_gainfo = 0
        
        # Load Conversiongain1.mat
        # data = loadmat('Conversiongain1.mat')
        # Conversiongain1 = data['Conversiongain1']
    
        if node1[0] == 0 and node1[1] == 0 and node1[2] <= 0:
            node_d = np.sqrt(node1[2]**2) + interrogator[2]  # distance when positioned vertically
        elif node1[0] == 0 and node1[1] == 0 and node1[2] > 0:
            node_d = interrogator[2] - np.sqrt(node1[2]**2)
        else:
            node_d = np.sqrt((interrogator[0] - node1[0])**2 +
                             (interrogator[1] - node1[1])**2 +
                             (interrogator[2] - node1[2])**2)
        
        # Calculate path loss for fo and second harmonic
        path_lossfo = -self.pathloss(node_d, fo)  # in dB
        path_loss2fo = -self.pathloss(node_d, second_harmonic)  # in dB
        
        Power_at_diode = Pt + int_gainfo + path_lossfo + gain_fdrfo  # received power at the diode
        if Power_at_diode < -42 or Power_at_diode > 0:
            raise ValueError('The power levels are not consistent with conversion gain')
        
        # CG = 0
        for row in self.Conversiongain1:
            if -abs(Power_at_diode) + abs(row[0]) <= 0.5 and -abs(Power_at_diode) + abs(row[0]) >= -0.5:
                CG = row[1]
                break
        # CG = 0
        phase1 = ((2 * np.pi * fo) / c) * node_d
        phase2 = ((2 * np.pi * second_harmonic) / c) * node_d
        phase = phase1 + phase2
        
        
        power = Pt + int_gainfo + path_lossfo + gain_fdrfo + CG + gain_fdr2fo + path_loss2fo + int_gain2fo + PLF
        # Link budget to determine the power received by the interrogator

        return power, phase


    def power_calculated(self, nodes, int_list):
        """
        Calculate the power received and phase for each interrogator and node.
    
        Parameters:
        - nodes: list of node objects (each object should have attributes x, y, z, phi, theta, r)
        - int_list: list of interrogator objects (each object should have attributes x, y, z, phi, theta, r)
    
        Returns:
        - Power: Power for the particular position of the node in dBm.
        """
        # phasor_voltages = np.zeros(1, dtype=complex)
        Pt = 35  # Power Transmitted in dBm
        index = 0
        N_Int = 1 # len(int_list)  # Number of interrogators
        
        # N_nodes = 1 # len(node)  # Number of nodes
        # counting nodes number
        if isinstance(nodes, dict): # dictionary of a single node
            N_nodes = 1 # 
        elif isinstance(nodes, list): # list of dictionary. each dic for each node
            N_nodes = len(nodes)
        phasor_voltages = np.zeros(N_nodes, dtype=complex)
        
        for j in range(N_Int):
            int_pos = self.position(int_list['x'], int_list['y'], int_list['z'])  # Getting the interrogator position
            U_int = self.unit_vector(int_list['theta'], int_list['phi'], int_list['r'])
            
            for i in range(N_nodes):
                if N_nodes == 1:
                    if isinstance(nodes, dict):
                        node = nodes
                    elif isinstance(nodes, list):
                        node = nodes[0]
                    # node = nodes
                else:
                    node = nodes[i] # picking a dic from the list
                   
                node_pos = self.position(node['x'], node['y'], node['z'])  # Getting the node position
    
                ref_int_pos = self.position(int_pos[0] - node_pos[0], int_pos[1] - node_pos[1], int_pos[2] - node_pos[2])  # Reference interrogator position
                azimuth, elevation, r = self.cart2sph(ref_int_pos[0], ref_int_pos[1], ref_int_pos[2])
                ref_theta = np.pi / 2 - elevation  # Getting the angular direction to see the gain due to the pattern
                ref_phi = azimuth
    
                D = 4 * np.cos(ref_theta)  # Directivity considered from the antenna pattern
                D_db = 10 * np.log10(D)  # Finding the gain in dB
                Un = self.unit_vector(node['theta'], node['phi'], node['r'])  # Unit_vector for nodes
                Poldb = self.plf(Un, U_int)  # Loss due to the polarization of the node and interrogator
    
                int_pos_new = self.position(0, 0, 0)  # Looking at the interrogator now as the origin
                ref_node_pos = self.position(int_pos[0] - node_pos[0], int_pos[1] - node_pos[1], int_pos[2] - node_pos[2])  # Observation of node by interrogator
    
                azimuth_int, elevation_int, r_int = self.cart2sph(ref_node_pos[0], ref_node_pos[1], ref_node_pos[2])
                ref_theta_int = np.pi / 2 - elevation_int
                ref_phi_int = azimuth
    
                D_int = 4 * np.cos(ref_theta_int)  # Taken from the radiation pattern of the interrogator

                
                D_db_int = 10 * np.log10(D_int)  # Gain for the interrogator
    
                Pr_one, phase = self.distance_power_sim(Pt, node_pos[0], node_pos[1], node_pos[2],
                                                        int_pos[0], int_pos[1], int_pos[2],
                                                        D_db, D_db_int, Poldb)
                                                   
                impedance_antenna = 50
                Power_observed_watts = 10 ** ((Pr_one - 30) / 10)  # Conversion to watts
                voltage_2 = np.sqrt(2 * impedance_antenna * Power_observed_watts)  # Finding the incidence voltage magnitude
    
                phasor_voltages[index] = voltage_2 * np.exp(1j * phase)  # Adding the phase
                index += 1
    
        
        Total_Voltage = np.sum(phasor_voltages)
        
        Total_Power_dbm = 10 * np.log10((1/2 * np.abs(Total_Voltage) ** 2) / 50) + 30
        Power = Total_Power_dbm
        
        # return Power
        return Power, Total_Voltage # matlab code e just return Power silo
    
    
    
