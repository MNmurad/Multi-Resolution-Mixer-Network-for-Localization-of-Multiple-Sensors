# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:50:15 2025

@author: Murad
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from svgpathtools import svg2paths
from svgpath2mpl import parse_path


def plot_custom(num_nodes, estimated_array, true_array, x_label = [], y_label = [], path = [], name = [], title = []):
    nrows, ncols = int(np.ceil(num_nodes/2)), 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    n = 0
    for i in range(nrows):
        for j in range(ncols):
            n += 1
            if n > num_nodes:
                break
            axs[i, j].plot(estimated_array[:, 2*n-2 : 2*n], label = ['pred-x', 'pred-y'])
            axs[i, j].plot(true_array[:, 2*n-2 : 2*n], label = ['true-x', 'true-y'])
            axs[i, j].set_title('node:{}'.format(n))
            axs[i, j].set_xlabel(x_label)
            axs[i, j].set_ylabel(y_label)
            axs[i, j].legend()
    plt.rcParams.update({'font.size': 8})
    fig.suptitle('{}'.format(title))
    plt.tight_layout()
    os.makedirs(path, exist_ok = True)
    plt.savefig(path + name + '_' + y_label + '.png', dpi = 500)
    plt.show()
    # np.save('estimated_array.npy', estimated_array)
    # np.save('true_array.npy', true_array)
    

def plot_custom2(array, num_figures, title = [], sub_title = [], x_label = [], y_label = [], path = [], name = []):
    nrows, ncols = int(np.ceil(num_figures/2)), 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    axs = axs.reshape(nrows, ncols)
    n = 0
    for i in range(nrows):
        for j in range(ncols):
            n += 1
            if n > num_figures:
                break
            axs[i, j].plot(array[:, n-1]) #, label = 'estimated')
            axs[i, j].set_title('{}:{}'.format(sub_title, n))
            axs[i, j].set_xlabel(x_label)
            axs[i, j].set_ylabel(y_label)
            axs[i, j].legend()
    plt.rcParams.update({'font.size': 8})
    fig.suptitle('{}'.format(title))
    plt.tight_layout()
    os.makedirs(path, exist_ok = True)
    plt.savefig(path + name + '_' + title + '.png', dpi = 500)
    # plt.show()


def make_video(input_path = [], output_path = [], name = []):
    output_video = output_path + name
    
    images = []
    num_files = len(os.listdir(input_path))
    for n in range(num_files):
        file_name = str(n+1) + '_' + 'fig.png'
        if file_name.endswith('.png'):
            file_path = os.path.join(input_path, file_name)
            images.append(imageio.v2.imread(file_path))
    imageio.mimsave(output_video, images, fps=24, macro_block_size = 1)
    return


def save_parameters(my_dict, path, name, estimates, trues, loss = None):
    df_estimates = pd.DataFrame(estimates)
    df_trues = pd.DataFrame(trues)
    df_estimates.to_csv(path + 'estimates.csv')
    df_trues.to_csv(path + 'trues.csv')
    with open(path + name, 'w') as f:    
        for key, value in vars(my_dict).items():
            f.write(f'{key}: {value}\n')
        f.write(f'mse_loss: {loss}')


class Custom_marker:
    def __init__(self):
        self.node = self.build_custom_marker('./utils/node_icon.svg')
        self.interrogator = self.build_custom_marker('./utils/interrogator_icon.svg')
        
    def build_custom_marker(self, path_icon):
        marker_path, attributes = svg2paths(path_icon)
        path = ''
        for i in range(len(attributes)):
            path = path + attributes[i]['d']
        custom_marker = parse_path(path)
        custom_marker.vertices -= custom_marker.vertices.mean(axis=0)
        return custom_marker