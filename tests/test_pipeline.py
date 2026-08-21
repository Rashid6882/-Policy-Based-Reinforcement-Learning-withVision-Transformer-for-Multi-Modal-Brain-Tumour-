import unittest
import os
import torch
import numpy as np
from dataset import BraTS2DDataset
from environment import BrainLesionEnv, compute_iou_bbox, compute_dice_mask, compute_metrics
from model import DQNNetwork
from agent_dqn import DQNAgent, DoubleDQNAgent

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data_dir = r"BraTS2020_training_data/content/data"

    def test_metrics_calculation(self):
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        iou = compute_iou_bbox(box1, box2)
        self.assertAlmostEqual(iou, 1.0)

        mask = np.zeros((240, 240), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        dice = compute_dice_mask(box1, mask)
        self.assertAlmostEqual(dice, 1.0)

        metrics = compute_metrics(box1, mask, box2)
        self.assertAlmostEqual(metrics['dice'], 1.0)
        self.assertAlmostEqual(metrics['iou'], 1.0)
        self.assertAlmostEqual(metrics['sensitivity'], 1.0)
        self.assertAlmostEqual(metrics['precision'], 1.0)

    def test_model_forward(self):
        model = DQNNetwork(in_channels=4, box_dim=5, action_dim=7)
        dummy_roi = torch.randn(2, 4, 64, 64)
        dummy_box_feat = torch.randn(2, 5)
        out = model(dummy_roi, dummy_box_feat)
        self.assertEqual(out.shape, (2, 7))

    def test_environment_step(self):
        env = BrainLesionEnv()
        sample = {
            'image': np.random.randn(4, 240, 240).astype(np.float32),
            'mask': np.ones((240, 240), dtype=np.uint8),
            'gt_bbox': np.array([20, 20, 100, 100], dtype=np.float32)
        }
        obs = env.reset(sample)
        self.assertEqual(obs['roi'].shape, (4, 64, 64))
        self.assertEqual(obs['box_feat'].shape, (5,))

        next_obs, reward, done, info = env.step(1) # Move right
        self.assertEqual(next_obs['roi'].shape, (4, 64, 64))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_agent_training_step(self):
        device = 'cpu'
        dqn_agent = DQNAgent(device=device)
        ddqn_agent = DoubleDQNAgent(device=device)

        dummy_roi = torch.randn(4, 64, 64)
        dummy_box = torch.tensor([0.1, 0.1, 0.8, 0.8, 0.0])

        for _ in range(35):
            dqn_agent.memory.push(dummy_roi, dummy_box, 0, 1.0, dummy_roi, dummy_box, False)
            ddqn_agent.memory.push(dummy_roi, dummy_box, 0, 1.0, dummy_roi, dummy_box, False)

        loss_dqn = dqn_agent.train_step(batch_size=32)
        loss_ddqn = ddqn_agent.train_step(batch_size=32)

        self.assertGreaterEqual(loss_dqn, 0.0)
        self.assertGreaterEqual(loss_ddqn, 0.0)

if __name__ == '__main__':
    unittest.main()
