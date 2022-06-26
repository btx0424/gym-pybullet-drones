from collections import defaultdict
from gym_pybullet_drones.envs.BaseAviary import ActionType
import numpy as np
from typing import List, Tuple, Dict
from gym_pybullet_drones.utils import xyz2rpy, rpy2xyz
import pybullet as p

EPS = 1e-6

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
        
        if self.act_type == ActionType.VEL or self.act_type == ActionType.VEL_ALIGNED:
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

        if self.act_type == ActionType.VEL or self.act_type == ActionType.VEL_ALIGNED:
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
            raise NotImplementedError(self.act_type)
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
            object_indices: Dict, 
            obs_split_sections,
            speed=1, mix=0.4, act_type=ActionType.VEL_RPY_EULER):
        # self.predator_indexes = np.array(predator_indexes) - len(prey_indexes) - len(predator_indexes) - 1
        # self.prey_indexes = np.array(prey_indexes) - len(prey_indexes) - 1
        self.object_indices = object_indices
        self.obs_split_sections = obs_split_sections
        self.speed = speed
        self.mix = mix
        self.act_type = act_type
    
    def reset(self):
        self._last_actions = defaultdict(lambda: np.zeros(3, dtype=np.float32))

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        indices = states.keys()
        states = np.stack(list(states.values()))
        states = np.split(states, self.obs_split_sections[:-1], axis=-1)
        pos_predators = np.stack([states[predator] for predator in self.object_indices["predator"]], 1)[..., :3]
        pos_selves = states[-1][..., :3]
        pos_wall = pos_selves / (np.linalg.norm(pos_selves) + EPS) * np.sqrt(2)
        if "cylinder" in self.object_indices.keys():
            cylinders = np.stack([states[cylinder] for cylinder in self.object_indices["cylinder"]], 1) # (prey, n, 4)

        if self.act_type == ActionType.VEL or self.act_type == ActionType.VEL_ALIGNED:
            actions = np.zeros((len(indices), 4))
            target_vel = np.zeros((len(indices), 3))
            weights = np.exp(-np.linalg.norm(pos_selves-pos_predators, axis=-1, keepdims=True))
            target_vel += np.sum((pos_selves-pos_predators) * weights, 1) # (prey, predator, 3) -> (prey, 3)
            # pull up!
            target_vel[:, 2] += (pos_selves[:, 2] < 0.15) / (pos_selves[:, 2] + 0.05) 
            # get back!
            weights = 1 / (np.linalg.norm(pos_selves-pos_wall, axis=-1, keepdims=True)+0.5)
            target_vel += (pos_selves-pos_wall) * weights 
            if "cylinder" in self.object_indices.keys(): # infinite cylinder ...
                cylinder_pos = cylinders[..., :3]
                cylinder_r = cylinders[..., 3]
                distance = np.linalg.norm((pos_selves-cylinder_pos)[:2], axis=-1, keepdims=True) - cylinder_r
                distance[distance < 0] = 0
                weights = np.exp(-distance)
                # target_vel[:, :2] += np.sum((pos_selves-cylinder_pos)[..., :2] * weights, 1)
            actions[:, :3] = target_vel
            actions[:, 3] = self.speed
        else:
            raise NotImplementedError(self.act_type)
        return {idx: actions[i] for i, idx in enumerate(indices)}

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
