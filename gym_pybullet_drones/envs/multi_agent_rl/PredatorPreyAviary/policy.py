from collections import defaultdict
import numpy as np
from typing import List, Tuple, Dict
from gym_pybullet_drones.utils import xyz2rpy, rpy2xyz


class WayPointPolicy:
    def __init__(self, waypoints, obs_split_sections) -> None:
        self.waypoints = waypoints
        self.waypoint_cnt = 0
        self.obs_split_sections = obs_split_sections
    
    def reset(self) -> None:
        self.waypoint_cnt = 0

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for idx, state in states.items():
            agent_action = np.zeros(7, dtype=np.float32)
            state_self = np.split(state, self.obs_split_sections[:-1])[-1]
            waypoint = self.waypoints[self.waypoint_cnt]
            target_vel = waypoint - state_self[:3]
            target_rpy = xyz2rpy(target_vel, True)
            agent_action[:3] = target_vel
            agent_action[3] = 1.1
            agent_action[4:] = target_rpy
            if np.linalg.norm(target_vel) < 0.05: self.waypoint_cnt = (self.waypoint_cnt + 1) % len(self.waypoints)
            actions[idx] = agent_action
        return actions

class VelDummyPolicy:
    def __init__(self, obs_split_sections, vel=0.5) -> None:
        self.obs_split_sections = obs_split_sections
        self.vel = vel

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for idx, state in states.items():
            agent_action = np.zeros(7, dtype=np.float32)
            state_prey, state_self = np.split(state, self.obs_split_sections[:-1])[-2:]
            target_vel = state_prey[:3] - state_self[:3]
            target_vel /= np.abs(target_vel).max()
            target_rpy = xyz2rpy(target_vel, True)
            agent_action[:3] = target_vel
            agent_action[3] = self.vel
            agent_action[4:] = target_rpy
            actions[idx] = agent_action
        return actions    

class RulePreyPolicy:
    def __init__(self, 
            map_config,
            predator_indexes, 
            prey_indexes, 
            obs_split_sections,
            vel=1, mix=0.6):
        
        self.predator_indexes = np.array(predator_indexes) - len(prey_indexes) - len(predator_indexes) - 1
        self.prey_indexes = np.array(prey_indexes) - len(prey_indexes) - 1
        self.obs_split_sections = obs_split_sections
        self.vel = vel
        self.mix = mix
    
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
        return actions

class RulePredatorPolicy:
    """
    A policy that uses A* search to find a shortest path from each predator to a prey.
    """
    def __init__(self,
            map_config,
            predator_indexes, 
            prey_indexes, 
            obs_split_sections):
        raise NotImplementedError("Not correctly implemented yet.")
        self.map_config = map_config
        self.predator_indexes = predator_indexes
        self.prey_indexes = prey_indexes,
        self.obs_split_sections = obs_split_sections

        adj = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), axis=-1).reshape(-1, 3)
        self.adj = adj[~(adj == 0).all(1)]

        min_xyz = np.array(map_config["map"]["min_xyz"])
        max_xyz = np.array(map_config["map"]["max_xyz"])
        num_cells = 20
        cell_size = (max_xyz - min_xyz) / num_cells
        xx, yy, zz = (np.linspace(min_xyz, max_xyz, num_cells+1)[:-1] + cell_size/2).T
        self.cell_centers = np.stack(np.meshgrid(xx, yy, zz, indexing="ij"), -1)

        self.is_obstacle = np.zeros(self.cell_centers.shape[:3], dtype=bool)
        map_config["obstacles"]["box"] += [
            [min_xyz[0]-0.1, 0, 1, 0.1, max_xyz[1], 1],
            [0, max_xyz[1]+0.1, 1, max_xyz[0], 0.1, 1],
            [max_xyz[0]+0.1, 0, 1, 0.1, max_xyz[1], 1],
            [0, min_xyz[1]-0.1, 1, max_xyz[0], 0.1, 1]
        ]
        for box in map_config["obstacles"]["box"]:
            center, half_extent = np.split(np.array(box), 2)
            half_extent += 0.1
            occupation = np.logical_and(
                np.all(self.cell_centers >= np.floor((center-half_extent-min_xyz)/cell_size)*cell_size+min_xyz, axis=-1),
                np.all(self.cell_centers <= np.ceil((center+half_extent-min_xyz)/cell_size)*cell_size+min_xyz, axis=-1))
            self.is_obstacle[occupation] = True
        self.max_xyz = max_xyz
        self.min_xyz = min_xyz
        self.cell_size = cell_size

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        action = {}
        for idx, state in states.items():
            agent_action = np.zeros(7, dtype=np.float32)
            prey_state, self_state = np.split(state, self.obs_split_sections[:-1])[-2:]
            pos_prey, pos_self = prey_state[:3], self_state[:3]
            cell_prey, cell_self = self._pos_to_cell(pos_self*self.max_xyz), self._pos_to_cell(pos_prey*self.max_xyz)
            if cell_prey!=cell_self:
                path = self._search_path(cell_self, cell_prey)
                target_pos = self.cell_centers[path[1]] / self.max_xyz
                target_vel = target_pos - pos_self
                target_rpy = xyz2rpy(target_vel, True)
                agent_action[:3] = target_vel
                agent_action[3] = 0.5
                agent_action[4:] = target_rpy
            else:
                print("already in the same cell")
            action[idx] = agent_action
        return action

    def _search_path(self, source, target):
        pos, is_obstacle = self.cell_centers, self.is_obstacle
        h = lambda x: np.sum(np.abs(pos[target] - pos[x]))
        d = lambda a, b: np.linalg.norm(pos[a] - pos[b])
        openset = [(0, source)]
        came_from = {}
        g_score = defaultdict(lambda: np.inf)
        g_score[source] = 0

        while len(openset) > 0:
            f, current = heapq.heappop(openset)
            if current == target:
                path = [current]
                while current in came_from.keys():
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for neighbor in current + self.adj:
                neighbor = tuple(neighbor)
                try:
                    if is_obstacle[neighbor] : continue
                except IndexError: continue
                t = g_score[current] + d(current, neighbor)
                if t < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = t
                    f = t + h(neighbor)
                    if (f, neighbor) not in openset:
                        heapq.heappush(openset, (f, neighbor))
        return []
    
    def _pos_to_cell(self, pos):
        return tuple(((pos - self.min_xyz) / self.cell_size).astype(int))