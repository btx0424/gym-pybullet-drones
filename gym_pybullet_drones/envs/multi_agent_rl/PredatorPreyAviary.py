import gym
import numpy as np
import pybullet as p
import yaml
import os
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary, ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl import BaseMultiagentAviary
from gym_pybullet_drones.utils import xyz2rpy, rpy2xyz
from typing import Dict
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle

class PredatorPreyAviary(BaseMultiagentAviary):
    def __init__(self,
        num_predators: int=3,
        num_preys: int=1,
        fov: float=np.pi/2,
        *,
        map_config = None,
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
        self.predators = list(range(num_predators))
        self.preys = list(range(num_predators, num_predators+num_preys))

        self.observe_obstacles = observe_obstacles
        self.obstacles = {}

        map_config = map_config or os.path.join(os.path.dirname(__file__), "maps", "square.yaml")
        with open(map_config, 'r') as f:
            self.map_config = yaml.safe_load(f)
        for obstacle_type, obstacle_list in self.map_config['obstacles'].items():
            if obstacle_type == 'box':
                self.obstacles['box'] = np.split(np.array(obstacle_list), 2, axis=1)
        super().__init__(drone_model=drone_model,
                         num_drones=num_predators+num_preys,
                         physics=Physics.PYB,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         obs=obs,
                         act=ActionType.VEL_RPY,
                         episode_len_sec=episode_len_sec,
                         max_xyz=self.map_config["boundary"]["max"],
                         min_xyz=self.map_config["boundary"]["min"])
        self.num_agents = self.NUM_DRONES

        if 'box' in self.obstacles.keys():
            box_centers, half_extents = self.obstacles['box']
            box_centers, half_extents = self._clipAndNormalizeXYZ(box_centers)[0], self._clipAndNormalizeXYZ(half_extents)[0]
            self.obstacles['box'] = (box_centers, half_extents)

    def reset(self, init_xyzs=None, init_rpys=None):
        if init_xyzs is not None: self.INIT_XYZS = init_xyzs
        if init_rpys is not None: self.INIT_RPYS = init_rpys
        obs = super().reset()
        return obs
    
    def _observationSpace(self) -> spaces.Dict:
        # TODO@Botian: 1. how to split? 2. differentiate predators and preys
        self.obs_split_shapes = []
        if self.observe_obstacles: 
            self.obs_split_shapes.append([len(self.obstacles['box'][0]), 6])
        self.obs_split_shapes.append([self.NUM_DRONES, self.single_obs_size])
        self.obs_split_sections = np.cumsum(np.concatenate([[dim]*num for num, dim in self.obs_split_shapes]))
        shape = self.obs_split_sections[-1]
        low = -np.ones(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32)
        return spaces.Dict({i: spaces.Box(low, high) for i in range(self.NUM_DRONES)})

    def _computeObs(self):
        states = np.stack(list(super()._computeObs().values()))
        obs = {}
        if self.observe_obstacles:
            boxes = np.concatenate(self.obstacles['box'], axis=1).flatten()
        for i in self.predators:
            others = np.arange(self.NUM_DRONES) != i
            obs_states = []
            if self.observe_obstacles:
                obs_states.append(boxes)
            obs_states.append(states[others].flatten()) # other drones
            obs_states.append(states[i]) # self
            obs[i] = np.concatenate(obs_states)
        return obs

    def _computeReward(self):
        # TODO@Botian: add support for multiple preys...
        assert len(self.preys) == 1
        rayFromPositions = self.pos[:len(self.predators)]
        rayToPositions = np.tile(self.pos[-1], (len(self.predators), 1))
        assert rayFromPositions.shape==rayToPositions.shape
        d = rayToPositions-rayFromPositions
        distance = np.linalg.norm(d, axis=-1, keepdims=True)
        d /= distance
        ori = rpy2xyz(self.rpy[:-1])
        in_fov = (d * ori).sum(1) > np.cos(self.fov/2)
        hit_id = np.array([hit[0] for hit in p.rayTestBatch(
            rayFromPositions=rayFromPositions,
            rayToPositions=rayToPositions
        )])
        in_sight = ((hit_id==self.NUM_DRONES) & in_fov & (distance < 0.5)).astype(float)

        reward = np.zeros(self.NUM_DRONES)
        reward[self.predators] += in_sight.sum() / len(self.predators)
        reward[self.preys] += -in_sight.sum()

        # collision penalty
        for drone_id in self.DRONE_IDS:
            contact_points = p.getContactPoints(bodyA=drone_id)
            if len(contact_points) > 0:
                reward[drone_id-1] -= 1
        return reward

    def _addObstacles(self):
        super()._addObstacles()
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
            ax.set_title(f"step {self.step_counter}")
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
            *, 
            map_config = None,
            drone_model: DroneModel = DroneModel.CF2X, 
            freq: int = 120, 
            aggregate_phy_steps: int = 1, 
            gui=False, 
            episode_len_sec=5,
            observe_obstacles: bool=True):
        self.prey = num_predators
        super().__init__(num_predators, 1, fov, 
            map_config=map_config,
            drone_model=drone_model, freq=freq, 
            aggregate_phy_steps=aggregate_phy_steps, 
            gui=gui, episode_len_sec=episode_len_sec, 
            observe_obstacles=observe_obstacles)
        
        self.num_agents = len(self.predators)
        self.waypoints = self.map_config['prey']['waypoints']

    def _actionSpace(self) -> spaces.Dict:
        action_space = super()._actionSpace()
        return spaces.Dict({i: action_space[i] for i in self.predators})

    def _observationSpace(self) -> spaces.Dict:
        observation_space = super()._observationSpace()
        return spaces.Dict({i: observation_space[i] for i in self.predators})

    def _computeObs(self):
        obs = super()._computeObs()
        return {i: obs[i] for i in self.predators}

    def reset(self, init_xyzs=None, init_rpys=None):
        if init_xyzs is None: init_xyzs = self.INIT_XYZS
        if init_rpys is None: init_rpys = self.INIT_RPYS
        self._resetPrey()
        init_xyzs[-1] = self.waypoints[0]
        obs = super().reset(init_xyzs, init_rpys)
        return {i: obs[i] for i in self.predators}

    def step(self, actions: Dict[int, np.ndarray]):
        actions[self.prey] = self._preyAction()
        return super().step(actions)

    def _computeReward(self):
        reward = super()._computeReward()
        return reward[self.predators]

    def _computeDone(self):
        done = super()._computeDone()
        return done[self.predators]

    def _resetPrey(self):
        if np.random.random() > 0.5:
            self.waypoints = np.flipud(self.waypoints)
        self.waypoint_cnt = 0
    
    def _preyAction(self):
        waypoint = self.waypoints[self.waypoint_cnt]
        action = np.zeros(7)
        target_vel = waypoint - self.pos[self.prey]
        target_rpy = xyz2rpy(target_vel, True)
        action[:3] = target_vel
        action[3] = 1.5
        action[4:] = target_rpy
        if np.linalg.norm(target_vel) < 0.05: self.waypoint_cnt = (self.waypoint_cnt + 1) % len(self.waypoints)
        return action

    def dummyPolicy(self, states: Dict[int, np.ndarray]):
        actions = {}
        for idx, state in states.items():
            action = np.zeros(7, dtype=np.float32)
            state_prey, state_self = np.split(state, self.obs_split_sections[:-1])[-2:]
            target_vel = state_prey[:3] - state_self[:3]
            target_rpy = xyz2rpy(target_vel, True)
            action[:3] = target_vel
            action[3] = 0.1
            action[4:] = target_rpy
            actions[idx] = action
        return actions

if __name__ == "__main__":
    import imageio
    import os.path as osp
    from tqdm import tqdm
    env = PredatorAviary(aggregate_phy_steps=4, episode_len_sec=20)
    print(env.obs_split_shapes)
    print(env.obs_split_sections)
    # obs = env.reset()
    # assert env.observation_space.contains(obs), obs
    # action = env.action_space.sample()
    # assert env.action_space.contains(action)
    # obs, _, _, _ = env.step(action)
    # assert env.observation_space.contains(obs)

    init_xyzs = env.INIT_XYZS.copy()
    obs = env.reset()
    frames = []
    reward_total = 0
    for i in tqdm(range(env.MAX_STEPS)):
        action = env.dummyPolicy(obs)
        # action = env.action_space.sample()
        assert env.action_space.contains(action), f"{env.action_space} does not contain {action}"

        obs, reward, done, info = env.step(action)
        reward_total += sum(reward)
        if i % 6 == 0: frames.append(env.render("mini_map"))
        assert not np.any(done), done
    # assert np.all(done), f"{env.step_counter}, {env.MAX_STEPS} * {env.AGGR_PHY_STEPS}"
    imageio.mimsave(
        osp.join(osp.dirname(osp.abspath(__file__)), f"test_{env.__class__.__name__}_{reward_total}.gif"),
        ims=frames,
        format="GIF"
    )
    print(reward_total)
