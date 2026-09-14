import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from preprocessing.preprocessing_pipeline import BraTSDataPipeline, extract_crop
from .actions import Actions, NUM_ACTIONS
from .state import update_bbox
from .rewards import compute_dice, calculate_reward

class BrainLesionEnv(gym.Env):
    """
    Gymnasium Environment for Multi-Modal Brain Lesion Localization.
    Decoupled from RL Agent & Model implementation.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir, case_ids, max_steps=20, subregion="WT",
                 randomize_initial_window=True, initial_scale=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.case_ids = case_ids
        self.max_steps = max_steps
        self.subregion = subregion
        self.randomize_initial_window = randomize_initial_window
        self.initial_scale = initial_scale
        self.pipeline = BraTSDataPipeline(data_dir)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-10.0, high=10.0, shape=(4, 64, 64), dtype=np.float32),
            "coords": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        })

        self.current_case_id = None
        self.slice_2d = None
        self.gt_mask = None
        self.bbox = None
        self.step_count = 0
        self.curr_dice = 0.0
        self.trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if options and "case_id" in options:
            self.current_case_id = options["case_id"]
        else:
            self.current_case_id = random.choice(self.case_ids)

        self.slice_2d, self.gt_mask, _ = self.pipeline.process_case(self.current_case_id, self.subregion)
        self.step_count = 0

        # Initialize bounding box
        s = self.initial_scale
        if self.randomize_initial_window:
            cx = random.uniform(s / 2.0, 1.0 - s / 2.0)
            cy = random.uniform(s / 2.0, 1.0 - s / 2.0)
        else:
            cx, cy = 0.5, 0.5

        self.bbox = [cx - s / 2.0, cy - s / 2.0, cx + s / 2.0, cy + s / 2.0]
        self.curr_dice = compute_dice(self.bbox, self.gt_mask)
        self.trajectory = [{
            "step": 0,
            "bbox": list(self.bbox),
            "dice": self.curr_dice,
            "action": None
        }]

        crop_tensor = extract_crop(self.slice_2d, self.bbox)
        obs = {
            "image": crop_tensor.astype(np.float32),
            "coords": np.array(self.bbox, dtype=np.float32)
        }
        info = {
            "case_id": self.current_case_id,
            "dice": self.curr_dice,
            "bbox": list(self.bbox)
        }
        return obs, info

    def step(self, action):
        self.step_count += 1
        prev_dice = self.curr_dice

        if action == Actions.TERMINATE:
            reward = calculate_reward(prev_dice, self.curr_dice, action)
            terminated = True
            truncated = False
        else:
            self.bbox = update_bbox(self.bbox, action)
            self.curr_dice = compute_dice(self.bbox, self.gt_mask)
            reward = calculate_reward(prev_dice, self.curr_dice, action)
            terminated = False
            truncated = (self.step_count >= self.max_steps)

        crop_tensor = extract_crop(self.slice_2d, self.bbox)
        obs = {
            "image": crop_tensor.astype(np.float32),
            "coords": np.array(self.bbox, dtype=np.float32)
        }

        self.trajectory.append({
            "step": self.step_count,
            "bbox": list(self.bbox),
            "dice": self.curr_dice,
            "action": int(action)
        })

        info = {
            "case_id": self.current_case_id,
            "dice": self.curr_dice,
            "bbox": list(self.bbox),
            "step": self.step_count,
            "trajectory": self.trajectory
        }

        return obs, reward, terminated, truncated, info
