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
from matplotlib.patches import Rectangle, Circle
import pdb

class OccupationAviary_empty(BaseMultiagentAviary):
    def __init__(self,
        num_predators: int=3,
        fov: float=np.pi/2,
        vision_range: float=np.inf,
        *,
        map_config = "mini_empty",
        drone_model: DroneModel=DroneModel.CF2X,
        freq: int=120,
        aggregate_phy_steps: int=1,
        gui=False,
        obs: ObservationType=ObservationType.KIN,
        episode_len_sec: int=5,
        observe_obstacles: bool=False,
        seed = 1,
        fixed_height = False,
        random_agent = False
        ):

        if obs not in [ObservationType.KIN, ObservationType.KIN20]:
            raise NotImplementedError(obs)

        if random_agent:
            num_agents = np.random.randint(2, num_predators)
        else:
            num_agents = num_predators

        self.fov = fov
        self.predators = list(range(num_agents))
        self.vision_range = vision_range
        self.fixed_height = fixed_height

        self.observe_obstacles = observe_obstacles
        self.obstacles = {}

        # set seed
        self.env_seed = seed

        # set goals
        self.num_goals = num_agents
        self.goal_size = 0.1 # absolute
        self.goals = np.zeros(shape=(self.num_goals, 4)) # relabsoluteative pos + size
        self.goals[:,3:] = self.goal_size

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

        self.avail = avail.nonzero()[0]
        self.grid_centers = centers / grid_shape * (max_xyz-min_xyz) + min_xyz

        super().__init__(drone_model=drone_model,
                         num_drones=num_agents,
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
        # compute for exe time
        self.global_max_steps = self.MAX_PHY_STEPS // self.AGGR_PHY_STEPS

        if 'box' in self.obstacles.keys():
            box_centers, half_extents = self.obstacles['box']
            box_centers, half_extents = \
                self._clipAndNormalizeXYZ(box_centers)[0], \
                self._clipAndNormalizeXYZ(half_extents)[0]
            self.obstacles['box'] = (box_centers, half_extents)

    def reset(self, init_task=None, init_xyzs="random", init_rpys=None):
        self.timestep = 0
        # self.INIT_XYZS : absolute pos
        if isinstance(init_xyzs, np.ndarray):
            self.INIT_XYZS = init_xyzs
            self.goals[:,0:3] = self.INIT_XYZS + 0.1
        elif init_xyzs == "random": 
            if self.fixed_height:
                self.grid_centers[:,2] = 1.0
            if not hasattr(self, "rng"): self.seed(seed=self.env_seed)
            sample_pos_idx = self.rng.choice(self.avail, self.NUM_DRONES, replace=False)
            self.INIT_XYZS = self.grid_centers[sample_pos_idx]
            # init goal
            sample_goal_idx = self.rng.choice(self.avail, self.num_goals, replace=False)
            self.goals[:,0:3] = self.grid_centers[sample_goal_idx]
        if init_rpys is not None: self.INIT_RPYS = init_rpys
        # set task by init_task
        if init_task is not None:
            self.INIT_XYZS = init_task[:self.num_agents * 3].reshape(self.num_agents,-1)
            self.goals[:,0:3] = init_task[self.num_agents * 3: (self.num_agents + self.num_goals) * 3].reshape(self.num_goals,-1)
        obs = super().reset()
        self.episode_reward = np.zeros(self.num_agents)
        self.collision_penalty = np.zeros(self.NUM_DRONES)
        self.alive = np.ones(self.NUM_DRONES, dtype=bool)

        # return task configuration
        task = []
        task.append(self.INIT_XYZS.reshape(-1))
        task.append(self.goals[:,0:3].reshape(-1))
        infos =[]
        for i in range(self.num_agents):
            infos.append({'tasks': np.concatenate(task)})
        return obs, infos
    
    def _observationSpace(self) -> spaces.Dict:
        # TODO@Botian: 1. how to split? 2. differentiate predators and preys
        self.obs_split_shapes = []
        if self.observe_obstacles: 
            self.obs_split_shapes.append([len(self.obstacles['box'][0]), 6]) # obstacles
        self.obs_split_shapes.append([self.goals.shape[0], self.goals.shape[1]]) # goals
        state_size = 12 if self.OBS_TYPE == ObservationType.KIN else 20
        self.obs_split_shapes.append([self.NUM_DRONES-1, state_size]) # other drones
        self.obs_split_shapes.append([1, state_size]) # self
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
        goals = np.concatenate(self.goals)

        # get info task
        self.step_task = []
        self.step_task.append(self.pos.reshape(-1))
        self.step_task.append(self.goals[:,0:3].reshape(-1))
        self.step_task = np.concatenate(self.step_task)

        # TODO@Botian: assym obs?
        for i in self.predators:
            others = np.arange(self.NUM_DRONES) != i
            obs_states = []
            if self.observe_obstacles:
                obs_states.append(boxes) # obstacles
            obs_states.append(goals) # goals
            obs_states.append(states[others].flatten()) # other drones
            obs_states.append(states[i]) # self
            obs[i] = np.concatenate(obs_states)
        return obs

    def _computeReward(self):
        drone_pos = self.pos[self.predators] # absolute
        goals_pos = self.goals[:,0:3] # np.max(self.MAX_XYZ)
        rewards = np.zeros(self.NUM_DRONES)
        success = 0
        success_reward = 0
        distance_reward = 0
        for i in range(self.num_goals):
            dists = [np.linalg.norm(drone_pos[j] - goals_pos[i]) for j in range(self.NUM_DRONES)]
            distance_reward -= min(dists)
            
            # success reward
            if min(dists) <= self.goal_size:
                success_reward += 10
                success += 1
        # share_reward
        rewards += success_reward
        rewards += distance_reward
        
        self.success = success / self.num_goals

        if success == self.num_goals:
            self.exe_time = min(self.timestep, self.exe_time)
        else:
            self.exe_time = self.global_max_steps

        # time penalty
        # rewards -= (self.timestep / self.global_max_steps) * 5

        # collision_penalty
        self.drone_collision = np.array([len(p.getContactPoints(bodyA=drone_id))>0 for drone_id in self.DRONE_IDS])
        rewards -= self.drone_collision.astype(np.float32) * 5
        
        self.episode_reward += rewards
        self.collision_penalty += self.drone_collision.astype(np.float32) * 5
        self.alive[self.collision_penalty > 100] = False
        return rewards

    def _computeDone(self):
        done = super()._computeDone()
        return done
        # return done | ~self.alive
    
    def _computeInfo(self):
        info = super()._computeInfo()
        for i in range(self.num_agents):
            info[i]["tasks"] = self.step_task.copy()
        return info

    # TODO@jiayu, rejection placement for obstacles and goals
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
        # add virtual goals
        # goals
        for center, half_extent in zip(self.goals[:,:3],self.goals[:,3]):
            visualShapeId = p.createVisualShape(
                shapeType=p.GEOM_SPHERE, 
                radius=half_extent,
                rgbaColor=[0.5,0.5,0.5,0.5])
            # collisionShapeId = p.createCollisionShape(
            #     shapeType=p.GEOM_BOX, halfExtents=half_extent*self.MAX_XYZ)
            p.createMultiBody(
                baseVisualShapeIndex=visualShapeId,
                basePosition=center)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.episode_reward += reward
        self.timestep = (self.timestep + 1) % self.global_max_steps
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
            # for center, half_extent in zip(*self.obstacles['box']):
            #     xy = (center[:2] - half_extent[:2]) * self.MAX_XYZ[:2]
            #     w, h = half_extent[:2] * 2 * self.MAX_XYZ[:2]
            #     ax.add_patch(Rectangle(xy, w, h))
            # goals
            for center, half_extent in zip(self.goals[:,:2],self.goals[:,3]):
                xy = center
                w = half_extent
                ax.add_patch(Circle(xy, w))
            ax.set_title(
                f"step {self.step_counter//self.AGGR_PHY_STEPS} "
                +f"drone_r: {np.sum(self.episode_reward[self.predators])} ")
            ax.set_xlim(self.MIN_XYZ[0], self.MAX_XYZ[0])
            ax.set_ylim(self.MIN_XYZ[1], self.MAX_XYZ[1])
            buffer, (width, height) = canvas.print_to_buffer()
            return np.frombuffer(buffer, np.uint8).reshape(height, width, -1)
        else:
            raise NotImplementedError(mode)

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
    
    def __call__(self, num_obstacles, num_goals, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for idx, state in states.items():
            agent_action = np.zeros(7, dtype=np.float32)
            state_one = np.split(state, self.obs_split_sections[:-1])
            state_self = state_one[-1][:3].copy()
            state_goal = np.array(state_one[num_obstacles:num_obstacles+num_goals])[:,0:3]
            dists = np.sum((state_goal - state_self)**2,axis=1)
            dists_index = np.argwhere(dists==np.min(dists))

            target_vel = (state_goal[dists_index] - state_self).squeeze(0)
            target_vel /= (np.abs(target_vel).max() + 1e-5)
            target_rpy = xyz2rpy(target_vel.squeeze(0), True)
            agent_action[:3] = target_vel
            agent_action[3] = 0.5
            agent_action[4:] = target_rpy
            actions[idx] = agent_action
        return actions

class VelZeroPolicy:
    def __init__(self, obs_split_sections) -> None:
        self.obs_split_sections = obs_split_sections
    
    def __call__(self, num_goals, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for idx, state in states.items():
            agent_action = np.zeros(7, dtype=np.float32)
            target_vel = np.zeros(3)
            target_rpy = xyz2rpy(target_vel, True)
            agent_action[:3] = 0.0
            agent_action[3] = -1.0
            agent_action[4:] = target_rpy
            actions[idx] = agent_action
        return actions

class DiscreteActionPolicy:
    def __init__(self, obs_split_sections, num_agents) -> None:
        self.obs_split_sections = obs_split_sections
        super().__init__()
        self.num_agents = num_agents
        self.action_space = [spaces.Discrete(27)] * self.num_agents
        self.directions = np.array(list(np.ndindex(3, 3, 3))) - 1

    def action(self, action):
        if isinstance(action, list): action = np.stack(action)
        vel_action = np.zeros((self.num_agents, 7))
        vel_action[:, :3] = np.squeeze(self.directions[action])
        vel_action[:, 3] = 0.5
        vel_action[:, 4:] = xyz2rpy(np.squeeze(self.directions[action]), True)
        return vel_action

if __name__ == "__main__":
    import imageio
    import os.path as osp
    from tqdm import tqdm
    num_drones = 2
    num_obstacles = 0
    env = OccupationAviary_empty(
        num_predators=num_drones,
        aggregate_phy_steps=4, episode_len_sec=15, 
        map_config="mini_empty",
        fixed_height=True)
    print(env.predators)
    print(env.obs_split_shapes)
    print(env.obs_split_sections)
    # obs = env.reset()
    # assert env.observation_space.contains(obs), obs
    # action = env.action_space.sample()
    # assert env.action_space.contains(action)
    # obs, _, _, _ = env.step(action)
    # assert env.observation_space.contains(obs)

    predator_policy = VelDummyPolicy(env.obs_split_sections)
    # predator_policy = DiscreteActionPolicy(env.obs_split_sections, num_drones)
    # predator_policy = VelZeroPolicy(env.obs_split_sections)
    # predator_policy = WayPointPolicy(
    #     env._clipAndNormalizeXYZ(env.map_config['prey']['waypoints'])[0], 
    #     env.obs_split_sections)

    init_xyzs = env.INIT_XYZS.copy()
    init_xyzs = np.array([[-1.0,-1.0,1.0],[-1.0,-1.0,0.5]])
    # init_xyzs[-1] = env.map_config['prey']['waypoints'][0]
    obs = env.reset(init_xyzs=init_xyzs)
    # obs = env.reset()
    frames = []
    reward_total = 0
    collision_penalty = 0
    for i in tqdm(range(env.MAX_PHY_STEPS//env.AGGR_PHY_STEPS)):
        action = {}
        # action.update(predator_policy({i: obs[i] for i in env.predators}))
        action.update(predator_policy(num_obstacles, num_drones, {i: obs[i] for i in env.predators}))
        # print('action', action)
        
        obs, reward, done, info = env.step(action)
        reward_total += sum(reward)
        # collision_penalty += sum(info[j]["collision_penalty"] for j in range(env.num_agents))
        # if i % 6 == 0: frames.append(env.render("mini_map"))
        if i % 6 == 0: frames.append(env.render("camera"))
        if np.all(done): break

    imageio.mimsave(
        osp.join(osp.dirname(osp.abspath(__file__)), f"test_camera_{env.__class__.__name__}_{reward_total}.gif"),
        ims=frames,
        format="GIF"
    )
    print(reward_total, collision_penalty, done)
