"""
visualize the pre-process of a action
"""
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from Action import *

FILENAME = "../data/rawdata/ntu/000001.fz"

class Visualize():
    def __init__(self, FILENAME):
        """
        create the visualization of a normed-pose of one frame
        """
        self.action = Action(FILENAME,25)
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)

    def update_norm_pose(self, i):
        """
        update the self.scatters
        """
        self.ax.clear()
        self.draw_norm_figure()
        data = np.array(self.action.norm_action_seq[i])
        scatters = self.ax.scatter(data[:,0],data[:,2],data[:,1])
        #self.scatters.set_3d_properties(data[:,2])

    def update_maxmin_pose(self, i):
        """
        update the self.scatters
        """
        self.ax.clear()
        self.draw_maxmin_figure()
        data = np.array(self.action.maxmin_action_seq[i])
        scatters = self.ax.scatter(data[:,0],data[:,2],data[:,1])
        vectors = np.array(self.action.pca_vector[i])
        origin = np.array(self.action.maxmin_MEAN[i])
        quivers = self.ax.quiver([origin[0],origin[0],origin[0]], [origin[2],origin[2],origin[2]], [origin[1], origin[1], origin[1]], vectors[:,0], vectors[:,2], vectors[:,1], colors=['r','g','b'], pivot='tail', arrow_length_ratio=0.1)
        #quivers = self.ax.quiver([0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1])
        #self.scatters.set_3d_properties(data[:,2])


    def draw_norm_figure(self):
        """
        create figure for visualize to show
        """
        self.ax.set_xlim3d([self.action.norm_bounder[0][0], self.action.norm_bounder[0][1]])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([self.action.norm_bounder[2][0], self.action.norm_bounder[2][1]])
        self.ax.set_ylabel('Z')
        self.ax.set_zlim3d([self.action.norm_bounder[1][0], self.action.norm_bounder[1][1]])
        self.ax.set_zlabel('Y')
        self.ax.set_title('norm action')


    def draw_maxmin_figure(self):
        """
        create figure for visualize to show
        """
        self.ax.set_xlim3d(0, 1)
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d(0, 1)
        self.ax.set_ylabel('Z')
        self.ax.set_zlim3d(0, 1)
        self.ax.set_zlabel('Y')
        self.ax.set_title('max-min action')


    def draw(self):

        ani = animation.FuncAnimation(self.fig, self.update_norm_pose, np.arange(0, self.action.frame_number), interval=25)
        plt.show()

    def draw_maxmin(self):
        ani = animation.FuncAnimation(self.fig, self.update_maxmin_pose, np.arange(0, self.action.frame_number), interval=25)
        plt.show()

    #def draw_cluster(self):

v = Visualize(FILENAME)
v.draw_maxmin()
