import gym
import numpy as np
import pybullet as p
import yaml
import os
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary, ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl import BaseMultiagentAviary
from gym_pybullet_drones.utils import xyz2rpy, rpy2xyz
from typing import Dict, List, Tuple
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from collections import defaultdict
import heapq

def in_sight_test(from_pos, to_pos, ori, fov, to_id, vision_range):
    d = to_pos[:, None] - from_pos # (n_prey, n_predators, 3)
    distance = np.linalg.norm(d, axis=-1, keepdims=True)
    d /= distance
    
    in_fov = ((d * ori).sum(-1) > np.cos(fov/2)) # 
    hit_id = np.array([hit[0] for hit in p.rayTestBatch(
        rayFromPositions=np.tile(from_pos.T, len(to_pos)).T.reshape(-1, 3),
        rayToPositions=np.tile(to_pos, len(from_pos)).reshape(-1, 3),
    )]).reshape(len(to_pos), len(from_pos)) - 1
    hit = (hit_id.T == np.array(to_id)).T

    in_sight = hit & in_fov & (distance.squeeze() < vision_range) # (prey, predators)
    return in_sight

class PredatorPreyAviary(BaseMultiagentAviary):
    def __init__(self,
        num_predators: int=3,
        num_preys: int=1,
        fov: float=np.pi/2,
        vision_range: float=np.inf,
        *,
        map_config = "square",
        drone_model: DroneModel=DroneModel.CF2X,
        freq: int=120,
        aggregate_phy_steps: int=1,
        gui=False,
        obs: ObservationType=ObservationType.KIN,
        episode_len_sec: int=5,
        observe_obstacles: bool=True
        ):

        if obs not in [ObservationType.KIN, ObservationType.KIN20]:
            raise NotImplementedError(obs)

        self.fov = fov
        self.num_predators = num_predators
        self.predators = list(range(num_predators))
        self.num_preys = num_preys
        self.preys = list(range(num_predators, num_predators+num_preys))
        self.vision_range = vision_range

        self.observe_obstacles = observe_obstacles
        self.obstacles = {}

        map_config = map_config or "square"
        if not map_config.endswith(".yaml"):
            map_config = os.path.join(os.path.dirname(__file__), "maps", f"{map_config}.yaml")
        with open(map_config, 'r') as f:
            self.map_config = yaml.safe_load(f)
        
        min_xyz = np.array(self.map_config["map"]["min_xyz"])
        max_xyz = np.array(self.map_config["map"]["max_xyz"])

        cell_size = 0.4
        grid_shape = ((max_xyz - min_xyz) / cell_size).astype(int)
        centers = np.array(list(np.ndindex(*(grid_shape)))) + 0.5
        avail = np.ones(len(centers), dtype=bool)

        for obstacle_type, obstacle_list in self.map_config['obstacles'].items():
            if obstacle_type == 'box':
                box_centers, half_extents = np.split(np.array(obstacle_list), 2, axis=1)
                for center, half_extent in zip(box_centers, half_extents):
                    min_corner = ((center-half_extent-min_xyz) / cell_size).astype(int)
                    max_corner = np.ceil((center+half_extent-min_xyz) / cell_size).astype(int)
                    mask = (centers > min_corner).all(1) & (centers < max_corner).all(1)
                    avail[mask] = False
                self.obstacles['box'] = (box_centers, half_extents)

        self.avail = avail.nonzero()[0]
        self.grid_centers = centers / grid_shape * (max_xyz-min_xyz) + min_xyz

        super().__init__(drone_model=drone_model,
                         num_drones=num_predators+num_preys,
                         physics=Physics.PYB,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         obs=obs,
                         act=ActionType.VEL_RPY,
                         episode_len_sec=episode_len_sec,
                         max_xyz=self.map_config["map"]["max_xyz"],
                         min_xyz=self.map_config["map"]["min_xyz"])
        self.num_agents = self.NUM_DRONES

        if 'box' in self.obstacles.keys():
            box_centers, half_extents = self.obstacles['box']
            box_centers, half_extents = \
                self._clipAndNormalizeXYZ(box_centers)[0], \
                self._clipAndNormalizeXYZ(half_extents)[0]
            self.obstacles['box'] = (box_centers, half_extents)

    def reset(self, init_xyzs=None, init_rpys=None):
        if isinstance(init_xyzs, np.ndarray):
            self.INIT_XYZS = init_xyzs
        elif init_xyzs == "random": 
            if not hasattr(self, "rng"): self.seed()
            sample_pos_idx = self.rng.choice(self.avail, self.NUM_DRONES, replace=False)
            self.INIT_XYZS = self.grid_centers[sample_pos_idx]
        if init_rpys is not None: self.INIT_RPYS = init_rpys
        obs = super().reset()
        self.episode_reward = np.zeros(self.num_agents)
        self.alive = np.ones(self.NUM_DRONES, dtype=bool)
        return obs
    
    def _observationSpace(self) -> spaces.Dict:
        # TODO@Botian: 1. how to split? 2. differentiate predators and preys
        self.obs_split_shapes = []
        if self.observe_obstacles: 
            self.obs_split_shapes.append([len(self.obstacles['box'][0]), 6])
        self.obs_split_shapes.append([self.NUM_DRONES-1, self.single_obs_size]) # other drones
        self.obs_split_shapes.append([1, self.single_obs_size]) # self
        self.obs_split_sections = np.cumsum(
            np.concatenate([[dim]*num for num, dim in self.obs_split_shapes]))
        shape = self.obs_split_sections[-1]
        low = -np.ones(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32)
        return spaces.Dict({i: spaces.Box(low, high) for i in range(self.NUM_DRONES)})

    def _computeObs(self):
        states = np.stack(list(super()._computeObs().values()))
        obs = {}
        if self.observe_obstacles:
            boxes = np.concatenate(self.obstacles['box'], axis=1).flatten()
        # TODO@Botian: assym obs?
        for i in self.predators + self.preys:
            others = np.arange(self.NUM_DRONES) != i
            obs_states = []
            if self.observe_obstacles:
                obs_states.append(boxes)
            obs_states.append(states[others].flatten()) # other drones
            obs_states.append(states[i]) # self
            obs[i] = np.concatenate(obs_states)
        return obs

    def _computeReward(self):
        rayFromPositions = self.pos[self.predators]
        rayToPositions = self.pos[self.preys]
        ori = rpy2xyz(self.rpy[self.predators]) # (n_predators, 3)
        in_sight = in_sight_test(rayFromPositions, rayToPositions, ori, self.fov, self.preys, self.vision_range)

        reward = np.zeros(self.NUM_DRONES)
        reward[self.predators] = np.sum(in_sight.any(1)) / self.num_predators
        reward[self.preys] = -np.sum(in_sight.any(1)) / self.num_preys
        assert reward.sum() == 0
        
        # collision_penalty
        self.drone_collision = np.array([len(p.getContactPoints(bodyA=drone_id))>0 for drone_id in self.DRONE_IDS])
        self.alive &= ~self.drone_collision
        reward -= self.drone_collision.astype(np.float32) * 5
        return reward
        # return reward * self.alive

    def _computeDone(self):
        done = super()._computeDone()
        return done
        # return done | ~self.alive
    
    def _computeInfo(self):
        info = super()._computeInfo()
        # for i in range(self.num_agents):
        #     info[i]["collision_penalty"] = -int(self.drone_collision[i])
        return info

    def _addObstacles(self):
        # add box obstacles
        if 'box' in self.obstacles.keys():
            for center, half_extent in zip(*self.obstacles['box']):
                visualShapeId = p.createVisualShape(
                    shapeType=p.GEOM_BOX, halfExtents=half_extent*self.MAX_XYZ)
                collisionShapeId = p.createCollisionShape(
                    shapeType=p.GEOM_BOX, halfExtents=half_extent*self.MAX_XYZ)
                p.createMultiBody(
                    baseCollisionShapeIndex=collisionShapeId,
                    baseVisualShapeIndex=visualShapeId,
                    basePosition=center*self.MAX_XYZ)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.episode_reward += reward
        return obs, reward, done, info

    def render(self, mode="camera"):
        if mode == "camera":
            return super().render()[0]
        elif mode == "mini_map":
            fig = Figure()
            canvas = FigureCanvasAgg(fig)
            ax = fig.gca()
            xy = self.pos[:, :2]
            uv = rpy2xyz(self.rpy)[:, :2]
            ax.quiver(*xy[self.predators].T, *uv[self.predators].T, color="r")
            ax.quiver(*xy[self.preys].T, *uv[self.preys].T, color="b")
            for center, half_extent in zip(*self.obstacles['box']):
                xy = (center[:2] - half_extent[:2]) * self.MAX_XYZ[:2]
                w, h = half_extent[:2] * 2 * self.MAX_XYZ[:2]
                ax.add_patch(Rectangle(xy, w, h))
            ax.set_title(
                f"step {self.step_counter//self.AGGR_PHY_STEPS} "
                +f"predator_r: {np.sum(self.episode_reward[self.predators])} "
                +f"prey_r: {np.sum(self.episode_reward[self.preys])} ")
            ax.set_xlim(self.MIN_XYZ[0], self.MAX_XYZ[0])
            ax.set_ylim(self.MIN_XYZ[1], self.MAX_XYZ[1])
            buffer, (width, height) = canvas.print_to_buffer()
            return np.frombuffer(buffer, np.uint8).reshape(height, width, -1)
        else:
            raise NotImplementedError(mode)

class PredatorAviary(PredatorPreyAviary):
    def __init__(self, 
            num_predators: int = 2, 
            fov: float = np.pi / 2, 
            vision_range: float=np.inf,
            *, 
            map_config = None,
            drone_model: DroneModel = DroneModel.CF2X, 
            freq: int = 120, 
            aggregate_phy_steps: int = 1, 
            gui=False, 
            episode_len_sec=5,
            observe_obstacles: bool=True):
        self.prey = num_predators
        super().__init__(num_predators, 1, fov, vision_range,
            map_config=map_config,
            drone_model=drone_model, freq=freq, 
            aggregate_phy_steps=aggregate_phy_steps, 
            gui=gui, episode_len_sec=episode_len_sec, 
            observe_obstacles=observe_obstacles)
        
        self.num_agents = len(self.predators)
        self.waypoints = np.array(self.map_config['prey']['waypoints'])

    def _actionSpace(self) -> spaces.Dict:
        action_space = super()._actionSpace()
        return spaces.Dict({i: action_space[i] for i in self.predators})

    def _observationSpace(self) -> spaces.Dict:
        observation_space = super()._observationSpace()
        return spaces.Dict({i: observation_space[i] for i in self.predators})

    def _computeObs(self):
        obs = super()._computeObs()
        self.prey_action = self.prey_policy({self.prey: obs[self.prey]})
        return {i: obs[i] for i in self.predators}

    def reset(self, init_xyzs=None, init_rpys=None):
        if init_xyzs is None: init_xyzs = self.INIT_XYZS
        if init_rpys is None: init_rpys = self.INIT_RPYS
        init_xyzs[-1] = self.waypoints[0]
        self.prey_policy = WayPointPolicy(
            self._clipAndNormalizeXYZ(self.waypoints)[0], self.obs_split_sections)
        obs = super().reset(init_xyzs, init_rpys)
        return {i: obs[i] for i in self.predators}

    def step(self, actions: Dict[int, np.ndarray]):
        actions.update(self.prey_action)
        return super().step(actions)

    def _computeReward(self):
        reward = super()._computeReward()
        return reward[self.predators]

    def _computeDone(self):
        done = super()._computeDone()
        return done[self.predators]

    def _computeInfo(self):
        info = super()._computeInfo()
        return info[self.predators]

    @staticmethod
    def dummyPolicy():
        return RulePredatorPolicy


class WayPointPolicy:
    def __init__(self, waypoints, obs_split_sections) -> None:
        self.waypoints = waypoints
        self.waypoint_cnt = 0
        self.obs_split_sections = obs_split_sections
    
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
    def __init__(self, obs_split_sections) -> None:
        self.obs_split_sections = obs_split_sections
    
    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for idx, state in states.items():
            agent_action = np.zeros(7, dtype=np.float32)
            state_prey, state_self = np.split(state, self.obs_split_sections[:-1])[-2:]
            target_vel = state_prey[:3] - state_self[:3]
            target_vel /= np.abs(target_vel).max()
            target_rpy = xyz2rpy(target_vel, True)
            agent_action[:3] = target_vel
            agent_action[3] = 0.1
            agent_action[4:] = target_rpy
            actions[idx] = agent_action
        return actions    

class RulePreyPolicy:
    def __init__(self, 
            predator_indexes, 
            prey_indexes, 
            obs_split_sections):
        raise NotImplementedError
        self.predator_indexes = predator_indexes
        self.prey_indexes = prey_indexes,
        self.obs_split_sections = obs_split_sections

    def __call__(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        for prey, state in states.items:
            state = np.split(state, self.obs_split_sections[:-1])
            pos_self = state[-1]
            pos_predators = state[self.predator_indexes]

        return

class RulePredatorPolicy:
    """
    A policy that uses A* search to find a shortest path from each predator to a prey.
    """
    def __init__(self,
            map_config,
            predator_indexes, 
            prey_indexes, 
            obs_split_sections):
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

def test(func):
    def foo(*args, **kwargs):
        print("Testing:", func.__name__)
        func(*args, **kwargs)
    return foo

@test
def test_in_sight():
    env = PredatorPreyAviary(
        num_predators=2, num_preys=4,
        aggregate_phy_steps=4, episode_len_sec=20, 
        map_config="square")

    init_xyzs = np.array([
        [0, 1, 0.3], 
        [0, -1, 0.3],
        [1, 1, 0.3],
        [1.1, 0, 0.3],
        [1, -1, 0.3],
        [0, 0, 0.3]
    ])
    obs = env.reset(init_xyzs=init_xyzs)
    print(in_sight_test(
        from_pos=env.pos[env.predators],
        to_pos=env.pos[env.preys],
        ori=rpy2xyz(env.rpy[env.predators]),
        fov=env.fov,
        to_id=env.preys,
        vision_range=env.vision_range
    ))
    env.close()

@test
def test_predator_rule_policy():
    env = PredatorPreyAviary(
        num_predators=1, num_preys=1,
        aggregate_phy_steps=4, episode_len_sec=20, 
        map_config="split")
    predator_policy = RulePredatorPolicy(env.map_config, env.predators, env.preys, env.obs_split_sections)
    
    obs = env.reset(init_xyzs="random")
    frames = []
    reward_total = 0
    try:
        for i in tqdm(range(env.MAX_PHY_STEPS//env.AGGR_PHY_STEPS)):
            action = {}
            action.update(predator_policy({i: obs[i] for i in env.predators}))
            action.update({i: np.zeros(7) for i in env.preys})
            
            obs, reward, done, info = env.step(action)
            reward_total += sum(reward)
            if np.all(done): break
            if i % 3 == 0: frames.append(env.render("mini_map"))
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        pass
    finally:
        imageio.mimsave(
            osp.join(osp.dirname(osp.abspath(__file__)), f"test_{env.__class__.__name__}_{reward_total}.gif"),
            ims=frames,
            format="GIF"
        )
        print(reward_total, done)

if __name__ == "__main__":
    import imageio
    import os.path as osp
    from tqdm import tqdm

    test_in_sight()
    test_predator_rule_policy()
