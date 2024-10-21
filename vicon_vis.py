'''This node also subscribes to Vicon position data but focuses purely on visualizing 
the position in 3D space without handling orientation.'''

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from vicon_receiver.msg import Position
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ViconSubscriber(Node):
    def __init__(self):
        super().__init__('vicon_subscriber')
        self.subscription = self.create_subscription(
            Position,
            '/vicon/Droplet2/Droplet2',
            self.position_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.current_position = None

        # Setting up the 3D plot
        plt.ion()  # Turn on interactive mode for live updates
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-2000, 2000)
        self.ax.set_ylim(-2000, 2000)
        self.ax.set_zlim(-2000, 2000)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')

    def position_callback(self, msg):
        self.current_position = (msg.x_trans, msg.y_trans, msg.z_trans)
        self.update_plot()

    def update_plot(self):
        if plt.fignum_exists(self.fig.number):
            self.ax.clear()
            self.ax.scatter(*self.current_position, color='b')
            self.ax.set_xlim(-2000, 2000)
            self.ax.set_ylim(-2000, 2000)
            self.ax.set_zlim(-2000, 2000)
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.set_zlabel('Z Position')
            plt.draw()
            plt.pause(0.01)  # Lower the pause to improve responsiveness

def main(args=None):
    rclpy.init(args=args)
    vicon_subscriber = ViconSubscriber()
    executor = SingleThreadedExecutor()

    executor.add_node(vicon_subscriber)

    try:
        while rclpy.ok() and plt.fignum_exists(vicon_subscriber.fig.number):
            executor.spin_once(timeout_sec=0.1)
            plt.pause(0.01)  # Integrate matplotlib processing into the ROS2 loop
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(vicon_subscriber.fig)  # Close the figure
        vicon_subscriber.destroy_node()  # Destroy the node
        rclpy.shutdown()  # Shutdown ROS

if __name__ == '__main__':
    main()
