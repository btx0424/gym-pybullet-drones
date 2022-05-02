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
        *,
        drone_model: DroneModel=DroneModel.CF2X,
        freq: int=240,
        aggregate_phy_steps: int=1,
        gui=False,
        obs: ObservationType=ObservationType.KIN,
        episode_len_sec=5,
        ):

        self.fov = fov
        self.predators = list(range(num_predators))
        self.preys = list(range(num_predators, num_predators+num_preys))

        super().__init__(drone_model=drone_model,
                         num_drones=num_predators+num_preys,
                         physics=Physics.PYB,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         obs=obs,
                         act=ActionType.VEL_RPY,
                         episode_len_sec=episode_len_sec)

    def reset(self, init_xyzs=None, init_rpys=None):
        if init_xyzs is not None: self.INIT_XYZS = init_xyzs
        if init_rpys is not None: self.INIT_RPYS = init_rpys
        obs = super().reset()
        return obs

    def _computeObs(self):
        obs = super()._computeObs()
        obs = {i: np.concatenate([obs[i], obs[self.NUM_DRONES-1]]) for i in self.predators}
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
        reward = {}
        reward.update({i: in_sight[i] for i in self.predators})
        reward.update({i: -in_sight.sum() for i in self.preys})
        return reward

class PredatorAviary(PredatorPreyAviary):
    def __init__(self, 
            num_predators: int = 2, 
            fov: float = np.pi / 2, 
            *, 
            drone_model: DroneModel = DroneModel.CF2X, 
            freq: int = 240, 
            aggregate_phy_steps: int = 1, 
            gui=False, 
            episode_len_sec=5):
        self.prey = num_predators
        super().__init__(num_predators, 1, fov, 
            drone_model=drone_model, freq=freq, 
            aggregate_phy_steps=aggregate_phy_steps, 
            gui=gui, episode_len_sec=episode_len_sec)

    def _observationSpace(self) -> spaces.Dict:
        # TODO@Botian: observe teammates' states
        low = -np.ones(20 * 2)
        high = np.ones(20 * 2)
        return spaces.Dict({i: spaces.Box(low, high) for i in self.predators})

    def reset(self, init_xyzs=None, init_rpys=None):
        obs = super().reset(init_xyzs, init_rpys)
        self._setPrey()
        return {i: obs[i] for i in self.predators}

    def step(self, actions: Dict[int, np.ndarray]):
        actions[self.prey] = self._preyAction()
        return super().step(actions)

    def _computeReward(self):
        reward = super()._computeReward()
        return {i: reward[i] for i in self.predators}

    def _computeDone(self):
        done = super()._computeDone()
        done.pop(self.prey)
        return done

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
        target_vel = waypoint - self.pos[self.prey]
        target_rpy = xyz2rpy(target_vel)
        if np.linalg.norm(target_vel) < 0.05: self.waypoint_cnt = (self.waypoint_cnt + 1) % len(self.waypoints)
        return np.concatenate([target_vel, target_rpy])

    @staticmethod
    def dummyPolicy(states: Dict[int, np.ndarray]):
        actions = {}
        for idx, state in states.items():
            state_self, state_prey = np.split(state, 2)
            target_vel = (state_prey[:3] - state_self[:3])
            target_rpy = xyz2rpy(target_vel)
            actions[idx] = np.concatenate([target_vel/np.linalg.norm(target_vel)*0.1, target_rpy])
        return actions

if __name__ == "__main__":
    import imageio
    import os.path as osp
    from tqdm import tqdm
    env = PredatorAviary()
    # obs = env.reset()
    # assert len(env.observation_space) == len(obs)
    # # assert env.observation_space.contains(obs)
    # action = env.action_space.sample()
    # assert env.action_space.contains(action)
    # obs, _, _, _ = env.step(action)
    # assert env.observation_space.contains(obs)

    init_xyzs = env.INIT_XYZS.copy()
    obs = env.reset(init_xyzs)
    frames = []
    reward_total = 0
    for i in tqdm(range(env.MAX_STEPS)):
        action = env.dummyPolicy(obs)
        obs, reward, done, info = env.step(action)
        reward_total += sum(reward.values())
        if i % 20 == 0: frames.append(env.render()[0])
    
    imageio.mimsave(
        osp.join(osp.dirname(osp.abspath(__file__)), f"test_{env.__class__.__name__}_{reward_total}.gif"),
        ims=frames,
        format="GIF"
    )
    print(reward_total)
