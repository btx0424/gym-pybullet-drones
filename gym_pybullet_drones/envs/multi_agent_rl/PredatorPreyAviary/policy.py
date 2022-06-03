from collections import defaultdict
from gym_pybullet_drones.envs.BaseAviary import ActionType
import numpy as np
from typing import List, Tuple, Dict
from gym_pybullet_drones.utils import xyz2rpy, rpy2xyz
import pybullet as p

class WayPointPolicy:
    def __init__(self, waypoints, obs_split_sections, act_type=ActionType.VEL_RPY_EULER, speed=1.) -> None:
        self.waypoints = waypoints
        self.waypoint_cnt = 0
        self.obs_split_sections = obs_split_sections
        self.act_type = act_type
        self.speed = speed
    
    def reset(self) -> None:
        self.waypoint_cnt = 0

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        indices = states.keys()
        states = np.stack(list(states.values()))

        state_self = np.split(states, self.obs_split_sections[:-1], axis=-1)[-1]
        pos_self = state_self[:, :3]
        pos_target = self.waypoints[self.waypoint_cnt]
        direction_vector = pos_target - pos_self
        
        if self.act_type == ActionType.VEL:
            actions = np.zeros((len(states), 4))
            actions[:, :3] = direction_vector
            actions[:, 3] = self.speed
        elif self.act_type == ActionType.VEL_RPY_EULER:
            actions = np.zeros((len(states), 7))
            actions[:, 3] = self.speed

            ## vector
            actions[:, :3] = direction_vector
            actions[:, 4:6] = direction_vector[:, :2] # leave z as 0

            ## euler angles
            # target_vel_rpy = xyz2rpy(direction_vector, True)
            # actions[:, :3] = target_vel_rpy
            # actions[:, -1] = target_vel_rpy[-1] # take only yaw
        elif self.act_type == ActionType.VEL_RPY_QUAT:
            actions = np.zeros((len(states), 9))
            actions[:, 4] = self.speed
            actions[:, :4] = np.stack([p.getQuaternionFromEuler(rpy) for rpy in xyz2rpy(direction_vector)])
            actions[:, 5:] = np.stack([p.getQuaternionFromEuler(rpy) for rpy in xyz2rpy(direction_vector)])
        else:
            raise NotImplementedError
        distance = np.linalg.norm(direction_vector, axis=-1)
        if np.any(distance < 0.05): 
            self.waypoint_cnt = (self.waypoint_cnt + 1) % len(self.waypoints)
        return {idx: actions[i] for i, idx in enumerate(indices)}   

class VelDummyPolicy:
    def __init__(self, obs_split_sections, speed=0.5, act_type=ActionType.VEL_RPY_EULER) -> None:
        self.obs_split_sections = obs_split_sections
        self.speed = speed
        self.act_type = act_type

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        indices = states.keys()
        states = np.stack(list(states.values()))

        state_prey, state_self = np.split(states, self.obs_split_sections[:-1], axis=-1)[-2:]
        pos_prey, pos_self = state_prey[:, :3], state_self[:, :3]
        direction_vector = pos_prey - pos_self

        if self.act_type == ActionType.VEL:
            actions = np.zeros((len(states), 4))
            actions[:, :3] = direction_vector
            actions[:, 3] = self.speed
        elif self.act_type == ActionType.VEL_RPY_EULER:
            actions = np.zeros((len(states), 7))
            actions[:, 3] = self.speed

            ## vector
            actions[:, :3] = direction_vector
            actions[:, 4:6] = direction_vector[:, :2] # leave z to 0

            ## euler angles
            # target_vel_rpy = xyz2rpy(direction_vector, True)
            # actions[:, :3] = target_vel_rpy
            # actions[:, -1] = target_vel_rpy[-1] # take only yaw
        elif self.act_type == ActionType.VEL_RPY_QUAT:
            actions = np.zeros((len(states), 9))
            actions[:, 4] = self.speed
            actions[:, :4] = np.stack([p.getQuaternionFromEuler(rpy) for rpy in xyz2rpy(direction_vector)])
            actions[:, 5:] = np.stack([p.getQuaternionFromEuler(rpy) for rpy in xyz2rpy(direction_vector)])
        else:
            raise NotImplementedError
        return {idx: actions[i] for i, idx in enumerate(indices)}   

class PredatorRulePolicy:
    def __init__(self,
            map_config,
            predator_indexes, 
            prey_indexes, 
            obs_split_sections,
            speed=1, mix=0.6, act_type=ActionType.VEL_RPY_EULER):

        self.predator_indexes = np.array(predator_indexes) - len(prey_indexes) - len(predator_indexes) - 1
        self.prey_indexes = np.array(prey_indexes) - len(prey_indexes) - 1
        self.obs_split_sections = obs_split_sections
        self.speed = speed
        self.mix = mix
        self.act_type = act_type
        

class RulePreyPolicy:
    def __init__(self, 
            map_config,
            predator_indexes, 
            prey_indexes, 
            obs_split_sections,
            speed=1, mix=0.6, act_type=ActionType.VEL_RPY_EULER):
        raise NotImplementedError
        self.predator_indexes = np.array(predator_indexes) - len(prey_indexes) - len(predator_indexes) - 1
        self.prey_indexes = np.array(prey_indexes) - len(prey_indexes) - 1
        self.obs_split_sections = obs_split_sections
        self.speed = speed
        self.mix = mix
        self.act_type = act_type
    
    def reset(self):
        self._last_actions = defaultdict(lambda: np.zeros(3, dtype=np.float32))

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for prey, state in states.items():
            state = np.split(state, self.obs_split_sections[:-1])
            pos_self = state[-1][:3]
            pos_predators = np.stack([state[predator] for predator in self.predator_indexes])[:, :3]
            pos_wall = pos_self / (np.linalg.norm(pos_self)+1e-6) * np.sqrt(2)
            d_squares = np.sum((pos_predators - pos_self)**2, axis=1, keepdims=True)
            agent_action = np.zeros(7, dtype=np.float32)
            target_vel = np.sum((pos_self - pos_predators) / (d_squares + 1e-6), axis=0)
            target_vel += (pos_self - pos_wall) / (np.sum((pos_wall - pos_self)**2+1e-6))
            target_vel += np.array([0, 0, 1/(pos_self[2]**2 + 1e-6)])
            target_vel += np.random.random(3) * 0.1
            target_vel = self.mix * target_vel + (1-self.mix) * self._last_actions[prey] # for smoothing
            agent_action[:3] = target_vel
            agent_action[3] = self.vel
            agent_action[4:] = xyz2rpy(target_vel, True)
            actions[prey] = agent_action
            self._last_actions[prey] = target_vel
        return actions

class PredatorDiscretePolicy:
    def __init__(self, obs_split_sections):
        self.obs_split_sections = obs_split_sections
        self.directions = np.array(list(np.ndindex(3, 3, 3))) - 1
        self.directions = self.directions / (np.linalg.norm(self.directions, axis=1, keepdims=True) + 1e-6)

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        indices = states.keys()
        states = np.stack(list(states.values()))

        state_prey, state_self = np.split(states, self.obs_split_sections[:-1], axis=-1)[-2:]
        pos_prey, pos_self = state_prey[:, :3], state_self[:, :3]
        direction_vector = pos_prey - pos_self
        distance = np.linalg.norm(direction_vector, axis=-1, keepdims=True)
        direction_vector /= distance

        actions = np.array((self.directions @ direction_vector.T).argmax(0))
        actions[distance.squeeze() < 0.1] = 13

        return {idx: actions[i] for i, idx in enumerate(indices)}
