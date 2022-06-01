import gym
import numpy as np
import pybullet as p
import yaml
import os
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary, ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl import BaseMultiagentAviary
from gym_pybullet_drones.envs.multi_agent_rl.PredatorPreyAviary import WayPointPolicy, VelDummyPolicy, RulePreyPolicy
from gym_pybullet_drones.utils import xyz2rpy, rpy2xyz
from typing import Dict, List, Tuple
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle, Circle

from termcolor import colored

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
        act: ActionType=ActionType.VEL_RPY_EULER,
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
        
        if self.map_config["map"]["wall"]: 
            if "box" not in self.map_config["obstacles"].keys():
                self.map_config["obstacles"]["box"] = []
            self.map_config["obstacles"]["box"] += [
                [min_xyz[0]-0.1, 0, 1, 0.1, max_xyz[1], 1],
                [0, max_xyz[1]+0.1, 1, max_xyz[0], 0.1, 1],
                [max_xyz[0]+0.1, 0, 1, 0.1, max_xyz[1], 1],
                [0, min_xyz[1]-0.1, 1, max_xyz[0], 0.1, 1]
            ]

        cell_size = 0.4
        grid_shape = ((max_xyz - min_xyz) / cell_size).astype(int)
        centers = np.array(list(np.ndindex(*(grid_shape)))) + 0.5
        avail = np.ones(len(centers), dtype=bool)

        if 'obstacles' in self.map_config.keys():
            for obstacle_type, obstacle_list in self.map_config["obstacles"].items():
                if obstacle_type == 'box':
                    box_centers, half_extents = np.split(np.array(obstacle_list), 2, axis=1)
                    for center, half_extent in zip(box_centers, half_extents):
                        min_corner = ((center-half_extent-min_xyz) / cell_size).astype(int)
                        max_corner = np.ceil((center+half_extent-min_xyz) / cell_size).astype(int)
                        mask = (centers > min_corner).all(1) & (centers < max_corner).all(1)
                        avail[mask] = False
                    self.obstacles['box'] = (box_centers, half_extents)
                if obstacle_type == 'cylinder':
                    cylinder_centers, radiuses = np.split(np.array(obstacle_list), [3], axis=1)
                    for center, radius in zip(cylinder_centers, radiuses):
                        min_corner = ((center-radius-min_xyz) / cell_size).astype(int)
                        max_corner = np.ceil((center+radius-min_xyz) / cell_size).astype(int)
                        mask = (centers > min_corner).all(1) & (centers < max_corner).all(1)
                        avail[mask] = False
                    self.obstacles['cylinder'] = (cylinder_centers, radiuses)

        self.avail = avail.nonzero()[0]
        self.grid_centers = centers / grid_shape * (max_xyz-min_xyz) + min_xyz

        super().__init__(drone_model=drone_model,
                         num_drones=num_predators+num_preys,
                         physics=Physics.PYB,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         obs=obs,
                         act=act,
                         episode_len_sec=episode_len_sec,
                         max_xyz=self.map_config["map"]["max_xyz"],
                         min_xyz=self.map_config["map"]["min_xyz"])
        self.num_agents = self.NUM_DRONES

        if 'obstacles' in self.map_config.keys():
            if 'box' in self.obstacles.keys():
                box_centers, half_extents = self.obstacles['box']
                box_centers, half_extents = \
                    self._clipAndNormalizeXYZ(box_centers)[0], \
                    self._clipAndNormalizeXYZ(half_extents)[0]
                self.obstacles['box'] = (box_centers, half_extents)
            if 'cylinder' in self.obstacles.keys():
                cylinder_centers, radiuses = self.obstacles['cylinder']
                cylinder_centers, radiuses = \
                    self._clipAndNormalizeXYZ(cylinder_centers)[0], \
                    radiuses / self.map_config["map"]["max_xyz"][0]
                self.obstacles['cylinder'] = (cylinder_centers, radiuses)

    def reset(self, init_xyzs="random", init_rpys=None):
        if isinstance(init_xyzs, np.ndarray):
            self.INIT_XYZS = init_xyzs
        elif init_xyzs == "random": 
            if not hasattr(self, "rng"): self.seed()
            sample_pos_idx = self.rng.choice(self.avail, self.NUM_DRONES, replace=False)
            self.INIT_XYZS = self.grid_centers[sample_pos_idx]
        if init_rpys is not None: self.INIT_RPYS = init_rpys
        obs = super().reset()
        self.episode_reward = np.zeros(self.NUM_DRONES)
        self.collision_penalty = np.zeros(self.NUM_DRONES)
        self.alive = np.ones(self.NUM_DRONES, dtype=bool)
        self.captured_steps = np.zeros(self.num_preys)
        return obs
    
    def _observationSpace(self) -> spaces.Dict:
        self.obs_split_shapes = [[1,1]] # time
        if self.observe_obstacles: 
            if 'box' in self.obstacles.keys(): self.obs_split_shapes.append([len(self.obstacles['box'][0]), 6])
            if 'cylinder' in self.obstacles.keys(): self.obs_split_shapes.append([len(self.obstacles['cylinder'][0]), 4])
        state_size = 12 if self.OBS_TYPE == ObservationType.KIN else 20
        self.obs_split_shapes.append([self.num_predators, state_size])
        self.obs_split_shapes.append([self.num_preys, state_size])
        self.obs_split_shapes.append([1, state_size])
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
            if 'box' in self.obstacles.keys(): boxes = np.concatenate(self.obstacles['box'], axis=1).flatten()
            if 'cylinder' in self.obstacles.keys(): cylinders = np.concatenate(self.obstacles['cylinder'], axis=1).flatten()
        for i in self.predators + self.preys:
            obs_states = [[self.step_counter / self.MAX_PHY_STEPS]]
            if self.observe_obstacles:
                if 'box' in self.obstacles.keys(): obs_states.append(boxes)
                if 'cylinder' in self.obstacles.keys(): obs_states.append(cylinders)
            obs_states.append(states[self.predators].flatten())
            obs_states.append(states[self.preys].flatten())
            obs_states.append(states[i])
            obs[i] = np.concatenate(obs_states)
        return obs

    def _computeReward(self):
        from_pos = self.pos[self.predators]
        to_pos = self.pos[self.preys]
        ori = rpy2xyz(self.rpy[self.predators]) # (n_predators, 3)

        d = to_pos[:, None] - from_pos # (n_prey, n_predators, 3)
        distance = np.linalg.norm(d, axis=-1, keepdims=True)
        # d /= distance
        
        # in_fov = ((d * ori).sum(-1) > np.cos(self.fov/2)) # 
        # hit_id = np.array([hit[0] for hit in p.rayTestBatch(
        #     rayFromPositions=np.tile(from_pos.T, len(to_pos)).T.reshape(-1, 3),
        #     rayToPositions=np.tile(to_pos, len(from_pos)).reshape(-1, 3),
        # )]).reshape(len(to_pos), len(from_pos)) - 1
        # hit = (hit_id.T == np.array(self.preys)).T

        # in_sight = hit & in_fov & (distance.squeeze() < self.vision_range) # (prey, predators)
        in_sight = np.ones((self.num_preys, self.num_predators), dtype=bool)
        
        reward = np.zeros(self.NUM_DRONES)
        min_distance = distance.min(1).flatten() # (num_prey,)
        reward_predators = (2*min_distance-0.2)*np.exp(-min_distance*3)*5
        reward_preys = np.exp(-min_distance)
        reward[self.predators] = np.sum(in_sight.any(1) * reward_predators) / self.num_predators
        reward[self.preys] = -np.sum(in_sight.any(1) * reward_preys) / self.num_preys
        
        # collision_penalty
        self.drone_collision = np.array([len(p.getContactPoints(bodyA=drone_id))>0 for drone_id in self.DRONE_IDS])
        reward -= self.drone_collision.astype(np.float32)

        self.episode_reward += reward
        self.collision_penalty += self.drone_collision.astype(np.float32)
        self.captured_steps[min_distance < 0.2] += 1
        self.alive[self.collision_penalty > 100] = False                                                         
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
        if 'cylinder' in self.obstacles.keys():
            for center, radius in zip(*self.obstacles['cylinder']):
                visualShapeId = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER, radius=radius*self.MAX_XYZ[0], length=self.MAX_XYZ[2])
                collisionShapeId = p.createCollisionShape(
                    shapeType=p.GEOM_CYLINDER, radius=radius*self.MAX_XYZ[0], height=self.MAX_XYZ[2])
                p.createMultiBody(
                    baseCollisionShapeIndex=collisionShapeId,
                    baseVisualShapeIndex=visualShapeId,
                    basePosition=center*self.MAX_XYZ)

    def step(self, action):
        obs, reward, done, info = super().step(action)
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
            if 'box' in self.obstacles.keys(): 
                for center, half_extent in zip(*self.obstacles['box']):
                    xy = (center[:2] - half_extent[:2]) * self.MAX_XYZ[:2]
                    w, h = half_extent[:2] * 2 * self.MAX_XYZ[:2]
                    ax.add_patch(Rectangle(xy, w, h))
            if 'cylinder' in self.obstacles.keys():
                for center, radius in zip(*self.obstacles['cylinder']):
                    ax.add_patch(Circle(center[:2] * self.MAX_XYZ[:2], radius*self.MAX_XYZ[0]))
            ax.set_title(
                f"step {self.step_counter//self.AGGR_PHY_STEPS} "
                +f"predator_r: {np.sum(self.episode_reward[self.predators]):.0f} "
                +f"prey_r: {np.sum(self.episode_reward[self.preys]):.0f} ")
            ax.set_xlim(self.MIN_XYZ[0], self.MAX_XYZ[0])
            ax.set_ylim(self.MIN_XYZ[1], self.MAX_XYZ[1])
            buffer, (width, height) = canvas.print_to_buffer()
            return np.frombuffer(buffer, np.uint8).reshape(height, width, -1)
        else:
            raise NotImplementedError(mode)

class PredatorAviary(PredatorPreyAviary):
    def __init__(self, 
            num_predators: int = 2, 
            num_preys: int = 1,
            fov: float = np.pi / 2, 
            vision_range: float=np.inf,
            prey_policy: str = "fixed",
            *, 
            map_config = None,
            drone_model: DroneModel = DroneModel.CF2X, 
            act: ActionType = ActionType.VEL_RPY_EULER,
            freq: int = 120, 
            aggregate_phy_steps: int = 1, 
            gui=False, 
            episode_len_sec=5,
            observe_obstacles: bool=True):
        self.prey = num_predators
        super().__init__(num_predators, num_preys, fov, vision_range,
            map_config=map_config,
            drone_model=drone_model, freq=freq, 
            aggregate_phy_steps=aggregate_phy_steps, 
            gui=gui, episode_len_sec=episode_len_sec, 
            observe_obstacles=observe_obstacles, act=act)
        
        self.num_agents = len(self.predators)
        self.waypoints = np.array(self.map_config['prey']['waypoints'])
        if prey_policy == "fixed":
            self.prey_policy = WayPointPolicy(
                self._clipAndNormalizeXYZ(self.waypoints)[0],
                self.obs_split_sections
            )
        elif prey_policy == "rule":
            self.prey_policy = RulePreyPolicy(
                self.map_config, self.predators, self.preys, self.obs_split_sections
            )

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

    def reset(self):
        if not hasattr(self, "rng"): self.seed()
        init_xyzs = self.INIT_XYZS.copy()
        init_xyzs[-1] = self.waypoints[0]
        sample_pos_idx = self.rng.choice(self.avail, self.num_predators, replace=False)
        init_xyzs[:-1] = self.grid_centers[sample_pos_idx]
        self.prey_policy.reset()
        obs = super().reset(init_xyzs)
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

def test(func):
    def foo(*args, **kwargs):
        print(colored("Testing:", "red"), func.__name__, kwargs)
        func(*args, **kwargs)
    return foo

@test
def test_in_sight():
    env = PredatorPreyAviary(
        num_predators=2, num_preys=4,
        aggregate_phy_steps=4, episode_len_sec=20, 
        map_config="square")
    print("obs_split_shapes", env.obs_split_shapes)
    print("obs_split_sections:", env.obs_split_sections)

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
def test_env():
    env = PredatorPreyAviary(
        num_predators=2, num_preys=2,
        aggregate_phy_steps=4, episode_len_sec=20, 
        map_config="split")
    print("obs_split_shapes", env.obs_split_shapes)
    print("obs_split_sections:", env.obs_split_sections)
    predator_policy = VelDummyPolicy(env.obs_split_sections)
    prey_policy = WayPointPolicy(
        env._clipAndNormalizeXYZ(np.array(env.map_config["prey"]["waypoints"]))[0],
        env.obs_split_sections)
    
    obs = env.reset(init_xyzs="random")
    frames = []
    reward_total = 0
    
    for i in tqdm(range(env.MAX_PHY_STEPS//env.AGGR_PHY_STEPS)):
        action = {}
        action.update(predator_policy({i: obs[i] for i in env.predators}))
        action.update(prey_policy({i: obs[i] for i in env.preys}))
        
        obs, reward, done, info = env.step(action)
        reward_total += sum(reward)
        if np.all(done): break
        if i % 4 == 0: frames.append(env.render("mini_map"))
    
    imageio.mimsave(
        osp.join(osp.dirname(osp.abspath(__file__)), f"test_{env.__class__.__name__}_{reward_total}.gif"),
        ims=frames,
        format="GIF"
    )
    print(env.pos)
    print(reward_total, done)

@test
def test_predator_aviary(prey_policy="fixed", act=ActionType.VEL_RPY_EULER):
    env = PredatorAviary(
        num_predators=2, num_preys=1,
        aggregate_phy_steps=4, episode_len_sec=20, 
        map_config="arena", prey_policy=prey_policy, act=act)
    print("obs_split_shapes", env.obs_split_shapes)
    print("obs_split_sections:", env.obs_split_sections)
    print("action_space:", env.action_space)
    predator_policy = VelDummyPolicy(env.obs_split_sections, speed=0.9, act_type=env.ACT_TYPE)
    
    obs = env.reset()
    frames = []
    reward_total = 0
    
    for i in tqdm(range(env.MAX_PHY_STEPS//env.AGGR_PHY_STEPS)):
        action = {}
        action.update(predator_policy({i: obs[i] for i in env.predators}))
        
        obs, reward, done, info = env.step(action)
        reward_total += sum(reward)
        if np.all(done): break
        if i % 4 == 0: frames.append(env.render("mini_map"))
    
    imageio.mimsave(
        osp.join(osp.dirname(osp.abspath(__file__)), f"test_{env.__class__.__name__}_{reward_total:.0f}.gif"),
        ims=frames,
        format="GIF"
    )
    print(reward_total, done)

if __name__ == "__main__":
    import imageio
    import os.path as osp
    from tqdm import tqdm

    # test_in_sight()
    # test_env()
    test_predator_aviary(prey_policy="fixed", act=ActionType.VEL)
    test_predator_aviary(prey_policy="fixed", act=ActionType.VEL_RPY_EULER)
    # test_predator_aviary("rule")

    # env = PredatorAviary(num_predators=1, num_preys=1,
    #     aggregate_phy_steps=4, episode_len_sec=20, 
    #     map_config="arena")
    # print(env.SPEED_LIMIT)
    # print(env.INIT_RPYS[0])
    # action = np.zeros(7)
    # action[:3] = np.array([1, 1, 1])
    # action[3] = 0.2
    # # action[4:] = xyz2rpy(np.array([-1, -1, 1]), True)
    # obs = env.reset()
    # for i in range(1000):
    #     obs, reward, done, info = env.step({i: action for i in env.predators})
    #     if i % 4 == 0: print(rpy2xyz(env.rpy[0]))