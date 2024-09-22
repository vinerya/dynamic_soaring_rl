import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class DynamicSoaringVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.trajectory = []

    def update(self, state):
        self.trajectory.append(state[:3])  # Only store position (x, y, z)

    def render(self):
        self.ax.clear()
        trajectory = np.array(self.trajectory)
        self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Dynamic Soaring Trajectory')
        plt.draw()
        plt.pause(0.001)

    def show(self):
        plt.show()

    def close(self):
        plt.close(self.fig)