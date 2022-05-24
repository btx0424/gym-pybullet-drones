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

class OccupationAviary(BaseMultiagentAviary):
    def __init__(self,
        num_drones: int=3,
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

        self.drones = list(range(num_drones))
        self.fov = fov
        self.vision_range = vision_range

        self.observe_obstacles = observe_obstacles
        self.obstacles = {}

        map_config = map_config or "square"
        if not map_config.endswith(".yaml"):
            map_config = os.path.join(os.path.dirname(__file__), "maps", f"{map_config}.yaml")
        with open(map_config, 'r') as f:
            self.map_config = yaml.safe_load(f)
        for obstacle_type, obstacle_list in self.map_config['obstacles'].items():
            if obstacle_type == 'box':
                self.obstacles['box'] = np.split(np.array(obstacle_list), 2, axis=1)
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
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
        self.episode_reward = np.zeros(self.num_agents)
        return obs
    
    def _observationSpace(self) -> spaces.Dict:
        # TODO@Botian: 1. how to split? 2. differentiate predators and preys
        self.obs_split_shapes = []
        if self.observe_obstacles: 
            self.obs_split_shapes.append([len(self.obstacles['box'][0]), 6])
        self.obs_split_shapes.append([self.NUM_DRONES-1, self.single_obs_size]) # other drones
        self.obs_split_shapes.append([1, self.single_obs_size]) # self
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
        # TODO@Botian: assym obs?
        for i in self.drones:
            others = np.arange(self.NUM_DRONES) != i
            obs_states = []
            if self.observe_obstacles:
                obs_states.append(boxes)
            obs_states.append(states[others].flatten()) # other drones
            obs_states.append(states[i]) # self
            obs[i] = np.concatenate(obs_states)
        return obs

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        # TODO@jiayu, set goals without physical bodies
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(int(self.NUM_DRONES/2)):
            val = -1 * np.linalg.norm(states[i, 0:3] - states[self.NUM_DRONES-1-i, 0:3])**2
            rewards[i] = val
            rewards[self.NUM_DRONES-1-i] = val

        # collision penalty
        
        self.drone_collision = 5 * np.array(
            [len(p.getContactPoints(bodyA=drone_id))>0 for drone_id in self.DRONE_IDS])\
            .astype(np.float32)
        rewards -= self.drone_collision
        return rewards

    def _computeInfo(self):
        info = super()._computeInfo()
        for i in range(self.num_agents):
            info[i]["collision_penalty"] = -self.drone_collision[i]
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
            ax.set_title(f"step {self.step_counter//self.AGGR_PHY_STEPS}, reward {self.episode_reward.sum()}")
            ax.set_xlim(self.MIN_XYZ[0], self.MAX_XYZ[0])
            ax.set_ylim(self.MIN_XYZ[1], self.MAX_XYZ[1])
            buffer, (width, height) = canvas.print_to_buffer()
            return np.frombuffer(buffer, np.uint8).reshape(height, width, -1)
        else:
            raise NotImplementedError(mode)

class VelDummyPolicy:
    def __init__(self, obs_split_sections) -> None:
        self.obs_split_sections = obs_split_sections
    
    def __call__(self, states: Dict[int, np.ndarray]):
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

if __name__ == "__main__":
    import imageio
    import os.path as osp
    from tqdm import tqdm
    env = OccupationAviary(aggregate_phy_steps=4, episode_len_sec=20, map_config="square")
    print(env.obs_split_shapes)
    print(env.obs_split_sections)
    # obs = env.reset()
    # assert env.observation_space.contains(obs), obs
    # action = env.action_space.sample()
    # assert env.action_space.contains(action)
    # obs, _, _, _ = env.step(action)
    # assert env.observation_space.contains(obs)

    policy = VelDummyPolicy(env.obs_split_sections)
    obs = env.reset()
    frames = []
    reward_total = 0
    collision_penalty = 0
    for i in tqdm(range(env.MAX_PHY_STEPS//env.AGGR_PHY_STEPS)):
        action = policy(obs)
        assert env.action_space.contains(action), f"{env.action_space} does not contain {action}"

        obs, reward, done, info = env.step(action)
        reward_total += sum(reward)
        collision_penalty += sum(info[j]["collision_penalty"] for j in range(env.num_agents))
        if i % 6 == 0: frames.append(env.render("mini_map"))
        assert not np.any(done), done
    # assert np.all(done), f"{env.step_counter}, {env.MAX_STEPS} * {env.AGGR_PHY_STEPS}"
    imageio.mimsave(
        osp.join(osp.dirname(osp.abspath(__file__)), f"test_{env.__class__.__name__}_{reward_total}.gif"),
        ims=frames,
        format="GIF"
    )
    print(reward_total, collision_penalty)
