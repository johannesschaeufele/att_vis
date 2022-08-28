import numpy as np

from skimage import io

import torch
import torch.nn as nn
from torchvision import transforms

from att_vis import AttentionVisualization


def read_image(path):
    image = io.imread(path).astype(np.float32) / 255.0

    return image


def get_sample_image():
    return read_image("assets/sample.png")


def normalize_l2(input):
    return input * torch.rsqrt(torch.sum(torch.square(input), dim=1, keepdim=True) + 1e-3)


def get_feature_maps(image):
    feature_dim = 64
    kernel_size = 5

    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        image_tensor = to_tensor(image).unsqueeze(0)

        layer_feature = nn.Sequential(nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(image_tensor.shape[1], feature_dim, kernel_size))

        conv_avg = nn.Conv2d(feature_dim, feature_dim, 3, groups=feature_dim)
        conv_avg.weight.data = torch.ones_like(conv_avg.weight.data) / feature_dim
        layer_avg = nn.Sequential(nn.ReflectionPad2d(3 // 2), conv_avg)

        feature_tensor = layer_feature(image_tensor)
        feature_tensor = feature_tensor - feature_tensor.mean((2, 3), keepdim=True)

        location_features = torch.randn_like(feature_tensor)
        location_features = normalize_l2(location_features)

        feature_tensor = feature_tensor + 0.5 * location_features

        feature_avg_tensor = layer_avg(feature_tensor)
        feature_avg_tensor = normalize_l2(feature_avg_tensor - feature_avg_tensor.mean((2, 3), keepdim=True))

        q_tensor = 10 * location_features
        k_tensor = 10 * feature_avg_tensor

        q = q_tensor.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        k = k_tensor.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

    return q, k


def get_sample_data():
    image = get_sample_image()

    qks = np.stack([np.stack(get_feature_maps(image)) for _ in range(3)])

    return qks


def main():
    qks = get_sample_data()

    vis = AttentionVisualization(qks)
    vis.show()


if __name__ == "__main__":
    main()
