# Assignment 1: Deep Learning Model Training & Analysis

## Colab Notebook Link
[Open in Google Colab](YOUR_COLAB_LINK_HERE)

---

## Q1(a): ResNet Classification on MNIST & FashionMNIST

### MNIST Results
| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16         | SGD       | 0.001         |                   |                   |
| 16         | SGD       | 0.0001        |                   |                   |
| ...        | ...       | ...           |                   |                   |

### FashionMNIST Results
| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16         | SGD       | 0.001         |                   |                   |
| ...        | ...       | ...           |                   |                   |

### Training Curves
![Loss Curve ResNet-18](./figures/loss_curve_resnet18.png)
![Accuracy Curve ResNet-50](./figures/acc_curve_resnet50.png)

---

## Q1(b): SVM Classifier Results
| Dataset       | Kernel | Accuracy (%) | Training Time (ms) |
|---------------|--------|--------------|---------------------|
| MNIST         | poly   |              |                     |
| MNIST         | rbf    |              |                     |
| FashionMNIST  | poly   |              |                     |
| FashionMNIST  | rbf    |              |                     |

---

## Q2: CPU vs GPU Performance Comparison

| Compute | Batch Size | Optimizer | LR    | ResNet-18 Acc | ResNet-32 Acc | ResNet-50 Acc | Train Time (ResNet-18) | FLOPs (ResNet-18) |
|---------|------------|-----------|-------|---------------|---------------|---------------|------------------------|-------------------|
| CPU     | 16         | SGD       | 0.001 |               |               |               |                        |                   |
| GPU     | 16         | SGD       | 0.001 |               |               |               |                        |                   |
| ...     | ...        | ...       | ...   |               |               |               |                        |                   |

### CPU vs GPU Comparison Chart
![CPU vs GPU Comparison](./figures/cpu_gpu_comparison.png)

---

## Analysis
(Add your detailed analysis here)
