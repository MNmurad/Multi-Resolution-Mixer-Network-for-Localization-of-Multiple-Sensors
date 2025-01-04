# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:08:32 2024

@author: Murad
"""

"""
Tutorials:
    1. https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
    2. 
    3. 
"""
import argparse
from sim.BAMS import BAMS, set_random_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Passive sensors localization')
    parser.add_argument('--task', type = str, default = 'ukf', choices = ['moving_dataset', 'ukf'])
    # Nodes and Interrogators positions
    parser.add_argument('--initial_nodes_position', type = float, nargs = '+', default = [-8.5, 10, -7, 10, -4.5, 10, -1.5, 10, 1, 9.5, 2.5, 9.3, 5, 9.8, 6.5, 9.8, 8.5, 9.4], help = "Initial positions of nodes (x, y) in a flat list.")
    parser.add_argument('--interrogators_position', type = float, nargs = '+', default = [-12, 12, 10, 1.571, 1.571, 1, 12, 12, 10, 1.571, 1.571, 1], help = "Interrogators positions (x, y, z, phi, theta, 1) in a flat list.")
    # Environment setup
    parser.add_argument('--x_boundary', type = float, nargs = '+', default = [-10, 10], help = 'x boundary of the experimental frame, [x_min, x_max]')
    parser.add_argument('--y_boundary', type = float, nargs = '+', default = [-10, 10], help = 'y boundary of the experimental frame, [y_min, y_max]')
    parser.add_argument('--slope_angle', type = float, default = 30.0, help = 'Slope angle in degree')
    parser.add_argument('--dt', type = float, default = 0.005, help = 'per step time for node')
    parser.add_argument('--seed', type = int, default = 42)
    # unscented kalman filter parameters
    parser.add_argument('--ukf_alpha', type = float, default = 1.0, help = 'alpha: (0, 1]')
    parser.add_argument('--ukf_beta', type = float, default = 2.0, help = 'in general beta = 2.0')
    parser.add_argument('--ukf_kappa', type = float, default = 0.01, help = 'kappa >=0; lambda = alpha^2 * (n + kappa) - n')
    parser.add_argument('--ukf_R_factor', type = float, default = 1e-8, help = 'Measurement Noise')
    parser.add_argument('--ukf_Q_factor', type = float, default = 1e-2, help = 'Process Noise')
    parser.add_argument('--ukf_P_factor', type = float, default = 1e-8, help = 'Initial Covariance, P: initial unceratainty of the position of the nodes')
    parser.add_argument('--path', type = str, default = './outputs/', help = 'output frames directory')
    # Special argument for data generation or UKF
    parser.add_argument('--velocity_model_id', type = str, default = 'test-1', choices = ['train', 'vali', 'test-1', 'test-2', 'test-3', 'test-4'], help = 'only used to  generate data. For ukf change the dataset inside the dataset folder.')
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    Bams = BAMS(args)
    
    if args.task == 'ukf':
        mse_loss = Bams.run()
        print(mse_loss)
        print('finished')
    elif args.task == 'moving_dataset':
        Bams.generate_datasets() # dataset for moving all nodes
