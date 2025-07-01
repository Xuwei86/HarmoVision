# HarmoVision

HarmoVision is a lightweight, high-performance CNN-Transformer hybrid model designed for efficient and accurate vision tasks, including image classification, object detection, and semantic segmentation. By integrating novel attention mechanisms—Local Window Multi-Head Self-Attention (LW-MHSA), Channel-Spatial Global Attention (CGSA), Directional Spatial Attention (DSA), and Dynamic Channel Attention (DCA)—HarmoVision achieves state-of-the-art performance while maintaining low computational complexity and parameter count. This repository contains the implementation of HarmoVision, along with scripts for training and evaluation on benchmark datasets like ImageNet, MS COCO, and ADE20K.

## Key Features

- **Efficient Architecture**: Combines CNN and Transformer strengths for balanced local and global feature extraction.
- **Attention Modules**:
  - **LW-MHSA**: Captures fine-grained local spatial dependencies with low computational overhead.
  - **CGSA**: Models global feature interactions and channel-wise importance using adaptive pooling and Swish activation.
  - **DSA & DCA**: Auxiliary modules that enhance spatial and channel attention, improving performance in downstream tasks.
- **State-of-the-Art Performance**: Outperforms lightweight models like MobileViTv1 and EMO on ImageNet, MS COCO, and ADE20K, as shown in ablation studies.
- **Lightweight Design**: Optimized for single-GPU training (e.g., NVIDIA 4090) with minimal parameters and FLOPs.
- **Flexible Implementation**: Built using the CVNets framework, supporting easy integration and experimentation.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NVIDIA GPU (e.g., 4090) with CUDA support
- CVNets framework (see [CVNets GitHub](https://github.com/apple/ml-cvnets))
- Datasets: ImageNet, MS COCO, ADE20K (prepared according to CVNets guidelines)

python train.py --config configs/imagenet_harmovision_s.yaml --dataset imagenet --model harmovision_s

1. **Project Overview**: Provides a concise description of HarmoVision, emphasizing its purpose, key components, and performance highlights to attract interest from researchers and developers.

2. **Key Features**: Highlights the innovative aspects of HarmoVision (e.g., LW-MHSA, CGSA, DSA, DCA) and its performance advantages, drawing from the manuscript’s results.

3. **Installation**: Includes clear instructions for setting up the environment, leveraging the CVNets framework (referenced in your manuscript) and standard dependencies for PyTorch-based projects.

