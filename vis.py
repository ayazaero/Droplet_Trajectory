'''A ROS node that subscribes to vehicle attitude data and visualizes the Euler angles 
(roll, pitch, yaw) derived from quaternion orientation data in a real-time plot.'''


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from px4_msgs.msg import VehicleAttitude
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class AttitudeSubscriber(Node):
    def __init__(self):
        super().__init__('attitude_subscriber')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

        self.fig, self.ax = plt.subplots()
        self.ln, = self.ax.plot([], [], 'ro')
        self.x_data, self.y_data, self.z_data = [], [], []
        self.ax.autoscale(enable=True)

    def quaternion_to_euler(self, q):
        # Converts quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)
        w, x, y, z = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def listener_callback(self, msg):
        quaternion = msg.q  # msg.q is an array of type float32[4]
        roll, pitch, yaw = self.quaternion_to_euler(quaternion)
        self.x_data.append(roll)
        self.y_data.append(pitch)
        self.z_data.append(yaw)

    def update_plot(self, frame):
        self.ln.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        return self.ln,

def main(args=None):
    rclpy.init(args=args)
    subscriber = AttitudeSubscriber()
    ani = FuncAnimation(subscriber.fig, subscriber.update_plot, blit=True, interval=50)
    plt.show()

    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
