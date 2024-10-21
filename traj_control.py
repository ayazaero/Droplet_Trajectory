'''Implements a model predictive control (MPC) node in ROS for trajectory planning. 
It subscribes to vehicle position and uses convex optimization to compute trajectory 
setpoints that are published to control the vehicle.'''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
import numpy as np
import cvxpy as cp
import time

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Define QoS settings based on publisher's profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,  # History is UNKNOWN from publisher, but KEEP_LAST is typically used for position data
            depth=1,  # Depth is not specified but generally set to 1 for real-time control applications
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos_profile)
        self.publisher = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10)
        self.command_publisher = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10)
        self.control_mode_publisher = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10)

        # MPC Setup
        self.T = 10  # Time horizon
        self.nx = 3  # State dimension
        self.nu = 3  # Control dimension
        self.dt = 0.1  # Time step
        self.Q = np.eye(self.nx)  # State cost matrix
        self.R = np.eye(self.nu)  # Control cost matrix

        self.current_state = np.zeros(self.nx)  # Initial state
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.offboard_set = False

    def position_callback(self, msg):
        # Check if the position is valid before updating
        if msg.xy_valid and msg.z_valid:
            # Use NED coordinates directly from the message
            self.current_state = np.array([msg.x, msg.y, msg.z])

    def timer_callback(self):
        if not self.offboard_set:
            self.engage_offboard_mode()
            self.arm_vehicle()
            self.offboard_set = True

        self.publish_offboard_control_mode()
        self.calculate_and_publish_trajectory_setpoint()

    def engage_offboard_mode(self):
        command = VehicleCommand()
        command.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        command.param1 = 1.0  # Arm
        command.param2 = 6.0  # Custom mode "offboard"
        command.timestamp = int(time.time() * 1_000_000)  # Convert to microseconds
        self.command_publisher.publish(command)

    def arm_vehicle(self):
        command = VehicleCommand()
        command.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        command.param1 = 1.0  # Arm
        command.timestamp = int(time.time() * 1_000_000)  # Convert to microseconds
        self.command_publisher.publish(command)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(time.time() * 1_000_000)
        self.control_mode_publisher.publish(msg)

    def calculate_and_publish_trajectory_setpoint(self):
        # Set up the optimization problem
        x = cp.Variable((self.T+1, self.nx))
        u = cp.Variable((self.T, self.nu))

        constraints = [x[0] == self.current_state]
        constraints += [x[k+1] == x[k] + self.dt * u[k] for k in range(self.T)]
        constraints += [cp.norm(u[k], 2) <= 0.1 for k in range(self.T)]

        objective = cp.Minimize(sum(cp.quad_form(x[k], self.Q) + cp.quad_form(u[k], self.R) for k in range(self.T)))

        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Extract the first control action
        optimal_velocity = np.array(u.value[0, :], dtype=np.float32)

        # Create and publish the TrajectorySetpoint message
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = int(time.time() * 1_000_000)
        trajectory_msg.velocity = optimal_velocity
        trajectory_msg.position = [float('nan')] * 3
        trajectory_msg.acceleration = [float('nan')] * 3
        trajectory_msg.jerk = [float('nan')] * 3
        trajectory_msg.yaw = float('nan')
        trajectory_msg.yawspeed = float('nan')

        self.publisher.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = MPCController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
