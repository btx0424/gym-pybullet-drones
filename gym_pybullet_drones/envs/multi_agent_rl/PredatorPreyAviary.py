import gym
import numpy as np
import pybullet as p
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary, ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl import BaseMultiagentAviary
from gym_pybullet_drones.utils import xyz2rpy, rpy2xyz
from typing import Dict

class PredatorPreyAviary(BaseMultiagentAviary):
    def __init__(self,
        num_predators: int=3,
        num_preys: int=1,
        fov: float=np.pi/2,
        num_obstacles: int=3,
        *,
        drone_model: DroneModel=DroneModel.CF2X,
        freq: int=240,
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

        self.num_obstacles = num_obstacles
        self.observe_obstacles = observe_obstacles

        super().__init__(drone_model=drone_model,
                         num_drones=num_predators+num_preys,
                         physics=Physics.PYB,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         obs=obs,
                         act=ActionType.VEL_RPY,
                         episode_len_sec=episode_len_sec)
        self.num_agents = self.NUM_DRONES

    def reset(self, init_xyzs=None, init_rpys=None):
        if init_xyzs is not None: self.INIT_XYZS = init_xyzs
        if init_rpys is not None: self.INIT_RPYS = init_rpys
        obs = super().reset()
        return obs
    
    def _observationSpace(self) -> spaces.Dict:
        # TODO@Botian: 1. how to split? 2. differentiate predators and preys
        self.obs_split_shapes = []
        if self.observe_obstacles: 
            self.obs_split_shapes.append([self.num_obstacles, 6])
        self.obs_split_shapes.append([self.NUM_DRONES, self.single_obs_size])
        self.obs_split_sections = np.cumsum(np.concatenate([[dim]*num for num, dim in self.obs_split_shapes]))
        shape = self.obs_split_sections[-1]
        low = -np.ones(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32)
        return spaces.Dict({i: spaces.Box(low, high) for i in range(self.NUM_DRONES)})

    def _computeObs(self):
        states = np.stack(list(super()._computeObs().values()))
        obs = {}
        for i in self.predators:
            others = np.arange(self.NUM_DRONES) != i
            obs_states = []
            if self.observe_obstacles:
                obs_states.append(np.concatenate(
                    [self.box_centers, self.half_extents], axis=1).flatten())
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
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        ori = rpy2xyz(self.rpy[:-1])
        in_fov = (d * ori).sum(1) > np.cos(self.fov/2)
        hit_id = np.array([hit[0] for hit in p.rayTestBatch(
            rayFromPositions=rayFromPositions,
            rayToPositions=rayToPositions
        )])
        in_sight = ((hit_id==self.NUM_DRONES) & in_fov).astype(float)
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
        box_centers =[]
        while len(box_centers) < self.num_obstacles:
            center = np.random.random(3)*2
            if not np.all(center[:2] < np.array([1, 1])):
                box_centers.append(center)
        self.box_centers = np.array(box_centers)
        self.half_extents = np.random.random((self.num_obstacles, 3))*0.1+0.1
        self.half_extents[:, -1] = self.box_centers[:, -1]
        
        for i in range(self.num_obstacles):
            visualShapeId = p.createVisualShape(
                shapeType=p.GEOM_BOX, halfExtents=self.half_extents[i])
            collisionShapeId = p.createCollisionShape(
                shapeType=p.GEOM_BOX, halfExtents=self.half_extents[i])
            p.createMultiBody(
                baseCollisionShapeIndex=collisionShapeId,
                baseVisualShapeIndex=visualShapeId,
                basePosition=self.box_centers[i])
        
        self.box_centers, _ = self._clipAndNormalizeXYZ(self.box_centers)
        self.half_extents, _ = self._clipAndNormalizeXYZ(self.half_extents)

class PredatorAviary(PredatorPreyAviary):
    def __init__(self, 
            num_predators: int = 2, 
            fov: float = np.pi / 2, 
            *, 
            drone_model: DroneModel = DroneModel.CF2X, 
            freq: int = 240, 
            aggregate_phy_steps: int = 1, 
            gui=False, 
            episode_len_sec=5,
            observe_obstacles: bool=True):
        self.prey = num_predators
        super().__init__(num_predators, 1, fov, 
            drone_model=drone_model, freq=freq, 
            aggregate_phy_steps=aggregate_phy_steps, 
            gui=gui, episode_len_sec=episode_len_sec, 
            observe_obstacles=observe_obstacles)

        self.num_agents = len(self.predators)

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
        obs = super().reset(init_xyzs, init_rpys)
        self._setPrey()
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

    def _setPrey(self):
        self.waypoints = np.array([
            [1, 0, 0.3], 
            [0, 1, 0.4], 
            [-1, 0, 0.2], 
            [0, -1, 0.3]
        ])
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

class DebugAviary(BaseMultiagentAviary):
    def __init__(self,
        num_agents=3,
        *,
        drone_model: DroneModel=DroneModel.CF2X,
        freq: int=240,
        aggregate_phy_steps: int=1,
        gui=False,
        obs: ObservationType=ObservationType.KIN,
        episode_len_sec=5,
        ):

        super().__init__(drone_model=drone_model,
                         num_drones=num_agents,
                         physics=Physics.PYB,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         obs=obs,
                         act=ActionType.VEL_RPY,
                         episode_len_sec=episode_len_sec)

        self.goal_state = np.zeros((self.NUM_DRONES, 20))
        self.goal_state[:, :3] += self.INIT_XYZS*2
        self.num_agents = self.NUM_DRONES

    def reset(self, init_xyzs=None, init_rpys=None):
        if init_xyzs is not None: self.INIT_XYZS = init_xyzs
        if init_rpys is not None: self.INIT_RPYS = init_rpys
        obs = super().reset()
        return obs

    def _observationSpace(self) -> spaces.Dict:
        shape = (self.NUM_DRONES+1) * self.single_obs_size
        low = -np.ones(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32)
        return spaces.Dict({i: spaces.Box(low, high) for i in range(self.NUM_DRONES)})
    
    def _computeObs(self):
        states = np.stack(list(super()._computeObs().values()))
        obs = {}
        for i in range(self.NUM_DRONES):
            others = np.arange(self.NUM_DRONES) != i
            obs[i] = np.concatenate([states[others], self.goal_state[i][None], states[i][None]]).reshape(-1)
        return obs
    
    def _computeReward(self):
        distance = np.linalg.norm(self.pos-self.goal_state[:, :3], axis=1)
        return -distance

    def dummyPolicy(self, states):
        actions = {}
        for idx in range(self.NUM_DRONES):
            action = np.zeros(7, dtype=np.float32)
            target_vel = self.goal_state[idx, :3] - self.pos[idx]
            target_rpy = xyz2rpy(target_vel, True)
            action[:3] = target_vel
            action[3] = min(1., np.linalg.norm(self.pos[idx]-self.goal_state[idx, :3]))
            action[4:] = target_rpy
            actions[idx] = action
        return actions

if __name__ == "__main__":
    import imageio
    import os.path as osp
    from tqdm import tqdm
    env = PredatorAviary(aggregate_phy_steps=5, episode_len_sec=10, observe_obstacles=False)
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
        if i % 10 == 0: frames.append(env.render()[0])
        assert not np.any(done), done
    # assert np.all(done), f"{env.step_counter}, {env.MAX_STEPS} * {env.AGGR_PHY_STEPS}"
    imageio.mimsave(
        osp.join(osp.dirname(osp.abspath(__file__)), f"test_{env.__class__.__name__}_{reward_total}.gif"),
        ims=frames,
        format="GIF"
    )
    print(reward_total)
