from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from uuv_mission.terrain import generate_reference_and_limits
import csv
import os
from uuv_mission.control import PDController  # 从 control 模块中导入 PDController

class Submarine:
    def __init__(self):
        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0

    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0

class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
    
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.references))
        min_depth = np.min(mission.cave_depths)
        max_height = np.max(mission.cave_heights)

        plt.fill_between(x_values, mission.cave_heights, mission.cave_depths, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depths, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_heights, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.references, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

class Mission:
    def __init__(self, references, cave_heights, cave_depths):
        self.references = references
        self.cave_heights = cave_heights
        self.cave_depths = cave_depths

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_path):
        references = []
        cave_heights = []
        cave_depths = []

        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                references.append(float(row['reference']))
                cave_heights.append(float(row['cave_height']))
                cave_depths.append(float(row['cave_depth']))

        return cls(references, cave_heights, cave_depths)

class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self, mission: Mission, disturbances: np.ndarray) -> Trajectory:
        T = len(mission.references)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()
            
            
            actions[t] = self.controller.control(mission.references[t], observation_t)
            
            
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.references))
        return self.simulate(mission, disturbances)

# Example usage:
# mission = Mission.from_csv('mission.csv')
# controller = PDController()  # 从 control.py 模块实例化 PDController
# submarine = Submarine()
# closed_loop = ClosedLoop(submarine, controller)
# disturbances = np.random.normal(0, 0.5, len(mission.references))
# trajectory = closed_loop.simulate(mission, disturbances)
# trajectory.plot_completed_mission(mission)