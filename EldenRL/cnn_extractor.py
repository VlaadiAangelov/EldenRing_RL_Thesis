import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiInputCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = 3

        # CNN for full frame
        self.img_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # CNN for player HP image
        self.hp_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # MLP input
        prev_action_dim = np.prod(observation_space['prev_actions'].shape)
        state_dim = observation_space['state'].shape[0]
        self.extra_input_dim = prev_action_dim + state_dim

        # Compute flattened CNN outputs
        with torch.no_grad():
            # Image sample
            sample_img = observation_space['img'].sample()
            if sample_img.shape[-1] != 3:  # Fix (C, H, W)
                sample_img = np.transpose(sample_img, (1, 2, 0))
            dummy_img = torch.from_numpy(sample_img).float().permute(2, 0, 1).unsqueeze(0)
            img_flatten = self.img_cnn(dummy_img).shape[1]

            # HP sample
            sample_hp = observation_space['boss_hp_img'].sample()
            if sample_hp.shape[-1] != 3:
                sample_hp = np.transpose(sample_hp, (1, 2, 0))
            dummy_hp = torch.from_numpy(sample_hp).float().permute(2, 0, 1).unsqueeze(0)
            hp_flatten = self.hp_cnn(dummy_hp).shape[1]

        # Final layer
        self.linear = nn.Sequential(
            nn.Linear(img_flatten + hp_flatten + self.extra_input_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x_img = observations['img'].float() / 255.0
        x_img = self.img_cnn(x_img)

        x_hp = observations['boss_hp_img'].float() / 255.0
        x_hp = self.hp_cnn(x_hp)

        x_extra = torch.cat([
            observations['prev_actions'].float().view(observations['prev_actions'].shape[0], -1),
            observations['state'].float()
        ], dim=1)

        x = torch.cat([x_img, x_hp, x_extra], dim=1)
        return self.linear(x)
