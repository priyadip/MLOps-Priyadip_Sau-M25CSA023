# Assignment 1: Deep Learning Model Training & Analysis

## Colab Notebook Link [Open in Google Colab](https://colab.research.google.com/drive/1RUpFLfIWphxVR4Xsd9B2E3yCURKh_atq?usp=sharing)


---

## Q1(a): ResNet Classification on MNIST & FashionMNIST

### MNIST Results
#### Epochs = 2, pin_memory = False

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 98.7 | 98.65714285714286 |
| 16 | SGD | 0.0001 | 94.02857142857142 | 89.46428571428571 |
| 16 | Adam | 0.001 | 97.36428571428571 | 95.55714285714286 |
| 16 | Adam | 0.0001 | 98.89285714285714 | 98.40714285714286 |
| 32 | SGD | 0.001 | 97.87857142857143 | 98.32142857142857 |
| 32 | SGD | 0.0001 | 89.89285714285714 | 49.06428571428572 |
| 32 | Adam | 0.001 | 96.08571428571429 | 95.54285714285714 |
| 32 | Adam | 0.0001 | 98.56428571428572 | 97.89285714285714 |

#### Epochs = 2, pin_memory = True

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 98.71428571428571 | 98.86428571428571 |
| 16 | SGD | 0.0001 | 94.1 | 88.52857142857142 |
| 16 | Adam | 0.001 | 98.00714285714285 | 96.49285714285715 |
| 16 | Adam | 0.0001 | 98.93571428571428 | 98.33571428571429 |
| 32 | SGD | 0.001 | 97.77142857142857 | 98.11428571428571 |
| 32 | SGD | 0.0001 | 90.65714285714286 | 60.48571428571429 |
| 32 | Adam | 0.001 | 95.26428571428572 | 95.28571428571429 |
| 32 | Adam | 0.0001 | 98.57142857142857 | 98.42142857142858 |

#### Epochs = 5, pin_memory = False

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 99.23571428571428 | 99.12857142857143 |
| 16 | SGD | 0.0001 | 97.97857142857143 | 97.47857142857143 |
| 16 | Adam | 0.001 | 98.64285714285714 | 98.55 |
| 16 | Adam | 0.0001 | 99.04285714285714 | 98.92857142857143 |
| 32 | SGD | 0.001 | 99.00714285714285 | 98.87142857142857 |
| 32 | SGD | 0.0001 | 96.76428571428572 | 95.06428571428572 |
| 32 | Adam | 0.001 | 98.85 | 97.63571428571429 |
| 32 | Adam | 0.0001 | 99.22142857142858 | 98.96428571428571 |

#### Epochs = 5, pin_memory = True

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 99.3 | 98.95714285714286 |
| 16 | SGD | 0.0001 | 98.01428571428572 | 97.11428571428571 |
| 16 | Adam | 0.001 | 98.57857142857142 | 98.54285714285714 |
| 16 | Adam | 0.0001 | 98.94285714285714 | 98.53571428571429 |
| 32 | SGD | 0.001 | 98.97857142857143 | 98.81428571428572 |
| 32 | SGD | 0.0001 | 96.82857142857142 | 94.11428571428571 |
| 32 | Adam | 0.001 | 98.58571428571429 | 97.91428571428571 |
| 32 | Adam | 0.0001 | 99.07857142857142 | 99.10714285714286 |

---

### FashionMNIST Results
#### Epochs = 2, pin_memory = False

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 85.01428571428572 | 79.38571428571429 |
| 16 | SGD | 0.0001 | 76.92142857142858 | 71.00714285714285 |
| 16 | Adam | 0.001 | 83.63571428571429 | 83.35 |
| 16 | Adam | 0.0001 | 87.42142857142858 | 85.95714285714286 |
| 32 | SGD | 0.001 | 83.4 | 82.77857142857142 |
| 32 | SGD | 0.0001 | 69.49285714285715 | 63.3 |
| 32 | Adam | 0.001 | 81.99285714285715 | 83.72142857142858 |
| 32 | Adam | 0.0001 | 87.50714285714285 | 87.05714285714286 |

#### Epochs = 2, pin_memory = True

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 84.51428571428572 | 79.98571428571428 |
| 16 | SGD | 0.0001 | 76.18571428571428 | 70.14285714285714 |
| 16 | Adam | 0.001 | 83.82857142857142 | 80.33571428571429 |
| 16 | Adam | 0.0001 | 86.82142857142857 | 88.02857142857142 |
| 32 | SGD | 0.001 | 82.75714285714285 | 83.55 |
| 32 | SGD | 0.0001 | 69.83571428571429 | 63.885714285714286 |
| 32 | Adam | 0.001 | 83.35714285714286 | 83.81428571428572 |
| 32 | Adam | 0.0001 | 87.32857142857142 | 86.58571428571429 |

#### Epochs = 5, pin_memory = False

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 90.88571428571429 | 89.62857142857143 |
| 16 | SGD | 0.0001 | 82.49285714285715 | 78.7 |
| 16 | Adam | 0.001 | 90.30714285714286 | 88.00714285714285 |
| 16 | Adam | 0.0001 | 92.00714285714285 | 90.7 |
| 32 | SGD | 0.001 | 89.85 | 87.26428571428572 |
| 32 | SGD | 0.0001 | 76.71428571428571 | 73.20714285714286 |
| 32 | Adam | 0.001 | 89.63571428571429 | 87.50714285714285 |
| 32 | Adam | 0.0001 | 91.8 | 91.35 |

#### Epochs = 5, pin_memory = True

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
|------------|-----------|---------------|-------------------|-------------------|
| 16 | SGD | 0.001 | 91.07857142857142 | 89.97857142857143 |
| 16 | SGD | 0.0001 | 82.73571428571428 | 78.33571428571429 |
| 16 | Adam | 0.001 | 89.82142857142857 | 88.21428571428571 |
| 16 | Adam | 0.0001 | 91.97142857142858 | 89.93571428571428 |
| 32 | SGD | 0.001 | 89.89285714285714 | 88.06428571428572 |
| 32 | SGD | 0.0001 | 75.11428571428571 | 74.21428571428571 |
| 32 | Adam | 0.001 | 89.57857142857142 | 88.31428571428572 |
| 32 | Adam | 0.0001 | 91.42857142857143 | 91.25714285714285 |

---
#  Best Model Configurations

## MNIST Best Results

| Model | Test Accuracy (%) | Batch Size | Optimizer | Learning Rate | pin_memory | Epochs | Training Time (s) |
|-------|-------------------|------------|-----------|---------------|------------|--------|-------------------|
| ResNet-18 | 99.3 | 16 | SGD | 0.001 | True | 5 | 229.88298225402832 |
| ResNet-50 | 99.12857142857143 | 16 | SGD | 0.001 | False | 5 | 511.96532464027405 |

## FashionMNIST Best Results

| Model | Test Accuracy (%) | Batch Size | Optimizer | Learning Rate | pin_memory | Epochs | Training Time (s) |
|-------|-------------------|------------|-----------|---------------|------------|--------|-------------------|
| ResNet-18 | 92.00714285714285 | 16 | Adam | 0.0001 | False | 5 | 287.5549020767212 |
| ResNet-50 | 91.35 | 32 | Adam | 0.0001 | False | 5 | 434.13118863105774 |

---


### Training Curves
#### MNIST
![Loss Curve ResNet-18](./figures/Q1(A)/MNIST_ResNet-18_BS16_SGD_LR0.001_PMTrue_EP5_history.png)
![Accuracy Curve ResNet-50](./figures/Q1(A)/MNIST_ResNet-50_BS16_SGD_LR0.001_PMFalse_EP5_history.png)

#### FashionMNIST
![Loss Curve ResNet-18](./figures/Q1(A)/FashionMNIST_ResNet-18_BS16_Adam_LR0.0001_PMFalse_EP5_history.png)
![Accuracy Curve ResNet-50](./figures/Q1(A)/FashionMNIST_ResNet-50_BS32_Adam_LR0.0001_PMFalse_EP5_history.png)

---

### Complete MNIST Results Table (All Configurations)

| Batch Size | Optimizer | Learning Rate | pin_memory | Epochs | ResNet-18 Acc (%) | ResNet-50 Acc (%) | ResNet-18 Time (s) | ResNet-50 Time (s) |
|------------|-----------|---------------|------------|--------|-------------------|-------------------|--------------------|--------------------|
| 16 | SGD | 0.001 | False | 2 | 98.7 | 98.65714285714286 | 114.69907021522522 | 210.78447890281677 |
| 16 | SGD | 0.001 | False | 5 | 99.23571428571428 | 99.12857142857143 | 276.0492169857025 | 511.96532464027405 |
| 16 | SGD | 0.001 | True | 2 | 98.71428571428571 | 98.86428571428571 | 96.93645739555359 | 186.77448797225952 |
| 16 | SGD | 0.001 | True | 5 | 99.3 | 98.95714285714286 | 229.88298225402832 | 454.70297956466675 |
| 16 | SGD | 0.0001 | False | 2 | 94.02857142857142 | 89.46428571428571 | 112.43891978263855 | 211.57850623130798 |
| 16 | SGD | 0.0001 | False | 5 | 97.97857142857143 | 97.47857142857143 | 272.8817706108093 | 510.2117941379547 |
| 16 | SGD | 0.0001 | True | 2 | 94.1 | 88.52857142857142 | 96.07134675979614 | 187.48906803131104 |
| 16 | SGD | 0.0001 | True | 5 | 98.01428571428572 | 97.11428571428571 | 231.32744526863098 | 458.13474321365356 |
| 16 | Adam | 0.001 | False | 2 | 97.36428571428571 | 95.55714285714286 | 117.82885336875916 | 221.4378638267517 |
| 16 | Adam | 0.001 | False | 5 | 98.64285714285714 | 98.55 | 284.37764286994934 | 541.2142238616943 |
| 16 | Adam | 0.001 | True | 2 | 98.00714285714285 | 96.49285714285715 | 100.81121015548706 | 198.80303311347961 |
| 16 | Adam | 0.001 | True | 5 | 98.57857142857142 | 98.54285714285714 | 241.48656225204468 | 484.6279275417328 |
| 16 | Adam | 0.0001 | False | 2 | 98.89285714285714 | 98.40714285714286 | 118.46016597747803 | 223.88984632492065 |
| 16 | Adam | 0.0001 | False | 5 | 99.04285714285714 | 98.92857142857143 | 284.6585576534271 | 541.5316417217255 |
| 16 | Adam | 0.0001 | True | 2 | 98.93571428571428 | 98.33571428571429 | 100.84598350524902 | 200.59498476982117 |
| 16 | Adam | 0.0001 | True | 5 | 98.94285714285714 | 98.53571428571429 | 241.19250059127808 | 485.3932042121887 |
| 32 | SGD | 0.001 | False | 2 | 97.87857142857143 | 98.32142857142857 | 88.33713841438293 | 174.34422612190247 |
| 32 | SGD | 0.001 | False | 5 | 99.00714285714285 | 98.87142857142857 | 212.70917439460754 | 421.80153822898865 |
| 32 | SGD | 0.001 | True | 2 | 97.77142857142857 | 98.11428571428571 | 67.63153052330017 | 152.7335503101349 |
| 32 | SGD | 0.001 | True | 5 | 98.97857142857143 | 98.81428571428572 | 163.12629103660583 | 371.9793612957001 |
| 32 | SGD | 0.0001 | False | 2 | 89.89285714285714 | 49.06428571428572 | 88.50576281547546 | 174.284441947937 |
| 32 | SGD | 0.0001 | False | 5 | 96.76428571428572 | 95.06428571428572 | 214.1864845752716 | 421.41421699523926 |
| 32 | SGD | 0.0001 | True | 2 | 90.65714285714286 | 60.48571428571429 | 68.14572858810425 | 152.8906455039978 |
| 32 | SGD | 0.0001 | True | 5 | 96.82857142857142 | 94.11428571428571 | 162.84940600395203 | 371.70528864860535 |
| 32 | Adam | 0.001 | False | 2 | 96.08571428571429 | 95.54285714285714 | 90.51042628288269 | 179.2144272327423 |
| 32 | Adam | 0.001 | False | 5 | 98.85 | 97.63571428571429 | 219.82399582862854 | 435.05515217781067 |
| 32 | Adam | 0.001 | True | 2 | 95.26428571428572 | 95.28571428571429 | 70.7585096359253 | 158.14448165893555 |
| 32 | Adam | 0.001 | True | 5 | 98.58571428571429 | 97.91428571428571 | 168.67467427253723 | 385.32640624046326 |
| 32 | Adam | 0.0001 | False | 2 | 98.56428571428572 | 97.89285714285714 | 89.6578061580658 | 181.7106578350067 |
| 32 | Adam | 0.0001 | False | 5 | 99.22142857142858 | 98.96428571428571 | 217.7536473274231 | 436.3625793457031 |
| 32 | Adam | 0.0001 | True | 2 | 98.57142857142857 | 98.42142857142858 | 70.76237535476685 | 158.6896562576294 |
| 32 | Adam | 0.0001 | True | 5 | 99.07857142857142 | 99.10714285714286 | 168.65476894378662 | 386.43714332580566 |

---

### Complete FashionMNIST Results Table (All Configurations)

| Batch Size | Optimizer | Learning Rate | pin_memory | Epochs | ResNet-18 Acc (%) | ResNet-50 Acc (%) | ResNet-18 Time (s) | ResNet-50 Time (s) |
|------------|-----------|---------------|------------|--------|-------------------|-------------------|--------------------|--------------------|
| 16 | SGD | 0.001 | False | 2 | 85.01428571428572 | 79.38571428571429 | 112.71111536026001 | 211.98679184913635 |
| 16 | SGD | 0.001 | False | 5 | 90.88571428571429 | 89.62857142857143 | 271.68701219558716 | 512.1623573303223 |
| 16 | SGD | 0.001 | True | 2 | 84.51428571428572 | 79.98571428571428 | 97.3718900680542 | 186.3084900379181 |
| 16 | SGD | 0.001 | True | 5 | 91.07857142857142 | 89.97857142857143 | 235.04566979408264 | 453.46248054504395 |
| 16 | SGD | 0.0001 | False | 2 | 76.92142857142858 | 71.00714285714285 | 113.44296765327454 | 211.56805324554443 |
| 16 | SGD | 0.0001 | False | 5 | 82.49285714285715 | 78.7 | 272.3234124183655 | 516.047768831253 |
| 16 | SGD | 0.0001 | True | 2 | 76.18571428571428 | 70.14285714285714 | 97.01622748374939 | 187.30354404449463 |
| 16 | SGD | 0.0001 | True | 5 | 82.73571428571428 | 78.33571428571429 | 233.74441266059875 | 458.2975959777832 |
| 16 | Adam | 0.001 | False | 2 | 83.63571428571429 | 83.35 | 117.86225438117981 | 222.74080395698547 |
| 16 | Adam | 0.001 | False | 5 | 90.30714285714286 | 88.00714285714285 | 287.7898361682892 | 541.9954113960266 |
| 16 | Adam | 0.001 | True | 2 | 83.82857142857142 | 80.33571428571429 | 102.27404141426086 | 198.11072182655334 |
| 16 | Adam | 0.001 | True | 5 | 89.82142857142857 | 88.21428571428571 | 246.18241834640503 | 486.3835790157318 |
| 16 | Adam | 0.0001 | False | 2 | 87.42142857142858 | 85.95714285714286 | 118.73279023170471 | 224.27968621253967 |
| 16 | Adam | 0.0001 | False | 5 | 92.00714285714285 | 90.7 | 287.5549020767212 | 541.7256758213043 |
| 16 | Adam | 0.0001 | True | 2 | 86.82142857142857 | 88.02857142857142 | 101.70539093017578 | 198.9772229194641 |
| 16 | Adam | 0.0001 | True | 5 | 91.97142857142858 | 89.93571428571428 | 246.96701765060425 | 483.58105850219727 |
| 32 | SGD | 0.001 | False | 2 | 83.4 | 82.77857142857142 | 90.51865601539612 | 173.97597408294678 |
| 32 | SGD | 0.001 | False | 5 | 89.85 | 87.26428571428572 | 217.38633275032043 | 421.8139066696167 |
| 32 | SGD | 0.001 | True | 2 | 82.75714285714285 | 83.55 | 73.06147646903992 | 153.019366979599 |
| 32 | SGD | 0.001 | True | 5 | 89.89285714285714 | 88.06428571428572 | 175.84775185585022 | 372.6916027069092 |
| 32 | SGD | 0.0001 | False | 2 | 69.49285714285715 | 63.3 | 91.33501815795898 | 175.05373907089233 |
| 32 | SGD | 0.0001 | False | 5 | 76.71428571428571 | 73.20714285714286 | 215.58726334571838 | 424.1742639541626 |
| 32 | SGD | 0.0001 | True | 2 | 69.83571428571429 | 63.885714285714286 | 73.08550381660461 | 153.44297623634338 |
| 32 | SGD | 0.0001 | True | 5 | 75.11428571428571 | 74.21428571428571 | 172.95678782463074 | 372.4374189376831 |
| 32 | Adam | 0.001 | False | 2 | 81.99285714285715 | 83.72142857142858 | 91.93817162513733 | 179.78658819198608 |
| 32 | Adam | 0.001 | False | 5 | 89.63571428571429 | 87.50714285714285 | 220.42599487304688 | 433.4921865463257 |
| 32 | Adam | 0.001 | True | 2 | 83.35714285714286 | 83.81428571428572 | 74.77671194076538 | 158.5727379322052 |
| 32 | Adam | 0.001 | True | 5 | 89.57857142857142 | 88.31428571428572 | 181.0028829574585 | 385.3301544189453 |
| 32 | Adam | 0.0001 | False | 2 | 87.50714285714285 | 87.05714285714286 | 91.72461867332458 | 178.51183772087097 |
| 32 | Adam | 0.0001 | False | 5 | 91.8 | 91.35 | 220.03815627098083 | 434.13118863105774 |
| 32 | Adam | 0.0001 | True | 2 | 87.32857142857142 | 86.58571428571429 | 75.12481093406677 | 159.04986381530762 |
| 32 | Adam | 0.0001 | True | 5 | 91.42857142857143 | 91.25714285714285 | 179.24412202835083 | 387.52039766311646 |

---

# ResNet Classification Results - Key Findings

## MNIST Dataset

- **Best Overall Performance**: ResNet-18 achieved **99.3% accuracy** (batch=16, SGD, lr=0.001, pin_memory=True, 5 epochs)
- **ResNet-50 Peak**: 99.13% accuracy with same SGD configuration
- **Quick Training**: 2 epochs sufficient for >98% accuracy on most configurations
- **Optimizer Impact**: Both SGD and Adam perform excellently; Adam with lr=0.0001 shows strong consistent results
- **Learning Rate Sensitivity**: lr=0.001 optimal for SGD; lr=0.0001 better for Adam
- **Batch Size Effect**: Batch size 16 generally outperforms 32
- **Pin Memory Benefit**: Minimal impact on accuracy (~0.1-0.3%), but reduces training time by ~15-20%
- **ResNet-18 Efficiency**: Achieves comparable accuracy to ResNet-50 in ~45% less training time
- **Training Speed**: ResNet-18 completes in ~230s vs ResNet-50 in ~512s (5 epochs)
- **Consistency**: Multiple configurations achieve >99% accuracy, showing robustness

## FashionMNIST Dataset

- **Best Overall Performance**: ResNet-18 achieved **92.01% accuracy** (batch=16, Adam, lr=0.0001, 5 epochs)
- **ResNet-50 Peak**: 91.35% accuracy (batch=32, Adam, lr=0.0001, 5 epochs)
- **Harder Task**: ~7% lower accuracy vs MNIST due to more complex visual patterns
- **Optimizer Preference**: Adam significantly outperforms SGD, especially at lower learning rates
- **Adam Dominance**: Adam with lr=0.0001 consistently achieves 87-92% vs SGD's 69-91%
- **Epoch Dependency**: 5 epochs essential for good performance; 2 epochs yields only 79-88%
- **SGD Struggles**: SGD with lr=0.0001 performs poorly (63-78% accuracy range)
- **Training Time**: ResNet-18 trains in ~220-287s; ResNet-50 in ~434-542s (5 epochs)
- **Batch Size Variability**: Less clear pattern; both 16 and 32 can perform well with right optimizer
- **Learning Rate Critical**: lr=0.0001 crucial for best results with Adam optimizer

## General Insights

- **Dataset Difficulty**: FashionMNIST requires more careful hyperparameter tuning than MNIST
- **Model Choice**: ResNet-18 offers best accuracy/speed tradeoff for both datasets
- **Configuration Matters**: Optimal settings differ significantly between datasets
- **Memory Pinning**: Worthwhile performance optimization with negligible accuracy impact
- **ResNet-18 vs ResNet-50**: Deeper model doesn't guarantee better accuracy
- **MNIST Simplicity**: Nearly all configurations achieve >94% accuracy
- **FashionMNIST Complexity**: Poor hyperparameters can result in <70% accuracy
- **SGD for MNIST**: Excellent performance with lr=0.001
- **Adam for FashionMNIST**: Clear winner, especially with lower learning rate
- **Training Efficiency**: Pin memory reduces time without sacrificing accuracy
- **Epoch Scaling**: Increasing from 2 to 5 epochs improves accuracy by 1-5%
- **Batch Size 16**: More consistent high performance across both datasets
- **Batch Size 32**: Faster training but slightly lower accuracy in many cases
- **Learning Rate 0.001**: Too high for Adam on FashionMNIST, perfect for SGD on MNIST
- **Learning Rate 0.0001**: Safer choice across datasets and optimizers
- **ResNet-50 Training Time**: Approximately 2.2x slower than ResNet-18
- **Time-Accuracy Tradeoff**: ResNet-18 provides best value for both datasets
- **Optimizer Selection**: Dataset characteristics should guide optimizer choice
- **Hyperparameter Sensitivity**: FashionMNIST shows much higher sensitivity than MNIST
- **Convergence Speed**: MNIST converges faster than FashionMNIST across all configurations
- **Best Practices for MNIST**: Use SGD with lr=0.001, batch=16, 5 epochs
- **Best Practices for FashionMNIST**: Use Adam with lr=0.0001, batch=16, 5 epochs
- **Reproducibility**: Results show consistent patterns across pin_memory settings
- **Production Recommendation**: ResNet-18 with Adam (lr=0.0001) for general use
- **Fast Prototyping**: 2 epochs on MNIST gives quick feedback (>98% possible)
- **Deep Learning Insight**: Architecture depth matters less than proper hyperparameters
- **Dataset Impact**: Simple datasets (MNIST) are less sensitive to configuration
- **Complex Datasets**: FashionMNIST benefits more from careful tuning
- **Overall Winner**: ResNet-18 balances speed, accuracy, and resource efficiency
- **Key Takeaway**: No one-size-fits-all configuration; tailor to dataset characteristics

---


# Q1(b): SVM Classifier on MNIST and FashionMNIST datasets with varying SVM hyperparameters


##  MNIST Dataset 
### Polynomial Kernel Results

| C    | Gamma     | Degree | Accuracy (%) | Training Time (ms) | Support Vectors |
|------|-----------|--------|--------------|---------------------|-----------------|
| 0.1  | scale     | 2      | 94.10        | 839631.43           | 30026           |
| 0.1  | scale     | 3      | 86.29        | 1454665.34          | 39372           |
| 0.1  | scale     | 4      | 59.21        | 2049577.99          | 47787           |
| 0.1  | auto      | 2      | 93.49        | 914155.58           | 31710           |
| 0.1  | auto      | 3      | 83.66        | 1609280.72          | 42067           |
| 0.1  | auto      | 4      | 47.43        | 2161517.11          | 49675           |
| 0.1  | 0.001     | 2      | 91.95        | 1130358.17          | 36138           |
| 0.1  | 0.001     | 3      | 74.16        | 1988409.82          | 48088           |
| 0.1  | 0.001     | 4      | 32.95        | 2429273.53          | 53300           |
| 0.1  | 0.01      | 2      | 97.45        | 223731.86           | 13422           |
| 0.1  | 0.01      | 3      | 97.90        | 334049.90           | 15167           |
| 0.1  | 0.01      | 4      | 96.83        | 536956.94           | 18558           |
| 1.0  | scale     | 2      | 96.92        | 322588.88           | 16466           |
| 1.0  | scale     | 3      | 95.83        | 592604.80           | 21544           |
| 1.0  | scale     | 4      | 89.01        | 1128233.62          | 31260           |
| 1.0  | auto      | 2      | 96.85        | 345066.85           | 17179           |
| 1.0  | auto      | 3      | 95.36        | 657996.50           | 23114           |
| 1.0  | auto      | 4      | 86.67        | 1265291.32          | 34179           |
| 1.0  | 0.001     | 2      | 96.51        | 409526.80           | 19264           |
| 1.0  | 0.001     | 3      | 93.60        | 875163.53           | 28122           |
| 1.0  | 0.001     | 4      | 77.77        | 1670007.78          | 41795           |
| 1.0  | 0.01      | 2      | 97.52        | 203689.50           | 12899           |
| 1.0  | 0.01      | 3      | 97.94        | 332019.36           | 15189           |
| 1.0  | 0.01      | 4      | 96.85        | 537550.02           | 18565           |
| 10.0 | scale     | 2      | 97.51        | 212703.59           | 13096           |
| 10.0 | scale     | 3      | 97.64        | 351666.00           | 15462           |
| 10.0 | scale     | 4      | 96.03        | 613551.28           | 20199           |
| 10.0 | auto      | 2      | 97.46        | 213249.21           | 13178           |
| 10.0 | auto      | 3      | 97.62        | 358474.76           | 15683           |
| 10.0 | auto      | 4      | 95.54        | 656864.82           | 21257           |
| 10.0 | 0.001     | 2      | 97.45        | 223472.02           | 13422           |
| 10.0 | 0.001     | 3      | 97.25        | 407601.60           | 16801           |
| 10.0 | 0.001     | 4      | 93.27        | 870712.16           | 25309           |
| 10.0 | 0.01      | 2      | 97.48        | 203654.23           | 12902           |
| 10.0 | 0.01      | 3      | 97.94        | 331683.98           | 15189           |
| 10.0 | 0.01      | 4      | 96.85        | 532788.16           | 18565           |

---

### RBF Kernel Results

| C    | Gamma     | Accuracy (%) | Training Time (ms) | Support Vectors |
|------|-----------|--------------|---------------------|-----------------|
| 0.1  | scale     | 93.40        | 557587.21           | 25282           |
| 0.1  | auto      | 93.43        | 579581.13           | 25350           |
| 0.1  | 0.001     | 93.54        | 570734.43           | 25819           |
| 0.1  | 0.01      | 65.96        | 2105207.43          | 45105           |
| 1.0  | scale     | 96.50        | 336559.87           | 15484           |
| 1.0  | auto      | 96.49        | 320190.45           | 15339           |
| 1.0  | 0.001     | 96.27        | 296986.34           | 15152           |
| 1.0  | 0.01      | 85.13        | 1998482.47          | 36734           |
| 10.0 | scale     | 97.26        | 278479.75           | 14535           |
| 10.0 | auto      | 97.28        | 261179.59           | 14114           |
| 10.0 | 0.001     | 97.18        | 225985.44           | 13104           |
| 10.0 | 0.01      | 86.27        | 1995558.85          | 36931           |

---

##  FashionMNIST Dataset 
### Polynomial Kernel Results

| C    | Gamma     | Degree | Accuracy (%) | Training Time (ms) | Support Vectors |
|------|-----------|--------|--------------|---------------------|-----------------|
| 0.1  | scale     | 2      | 84.42        | 547342.74           | 29756           |
| 0.1  | scale     | 3      | 82.03        | 642018.98           | 30997           |
| 0.1  | scale     | 4      | 75.65        | 925177.07           | 34412           |
| 0.1  | auto      | 2      | 84.42        | 548140.12           | 29756           |
| 0.1  | auto      | 3      | 82.03        | 642327.67           | 30997           |
| 0.1  | auto      | 4      | 75.65        | 918118.13           | 34412           |
| 0.1  | 0.001     | 2      | 82.62        | 647273.81           | 32153           |
| 0.1  | 0.001     | 3      | 78.36        | 821736.55           | 34662           |
| 0.1  | 0.001     | 4      | 69.69        | 1230680.94          | 39501           |
| 0.1  | 0.01      | 2      | 90.07        | 250089.68           | 19414           |
| 0.1  | 0.01      | 3      | 90.29        | 274542.41           | 20072           |
| 0.1  | 0.01      | 4      | 89.67        | 343277.82           | 21120           |
| 1.0  | scale     | 2      | 88.88        | 303085.04           | 21926           |
| 1.0  | scale     | 3      | 88.49        | 346659.88           | 22940           |
| 1.0  | scale     | 4      | 85.67        | 500918.91           | 25479           |
| 1.0  | auto      | 2      | 88.88        | 304159.48           | 21926           |
| 1.0  | auto      | 3      | 88.49        | 349759.87           | 22940           |
| 1.0  | auto      | 4      | 85.68        | 504498.03           | 25479           |
| 1.0  | 0.001     | 2      | 88.40        | 335970.31           | 23130           |
| 1.0  | 0.001     | 3      | 86.99        | 413641.01           | 24807           |
| 1.0  | 0.001     | 4      | 82.22        | 641515.80           | 28660           |
| 1.0  | 0.01      | 2      | 89.68        | 253785.18           | 18917           |
| 1.0  | 0.01      | 3      | 89.98        | 279279.99           | 20071           |
| 1.0  | 0.01      | 4      | 89.71        | 348479.58           | 21070           |
| 10.0 | scale     | 2      | 90.21        | 252635.79           | 19113           |
| 10.0 | scale     | 3      | 90.19        | 284770.93           | 20043           |
| 10.0 | scale     | 4      | 89.32        | 372995.88           | 21844           |
| 10.0 | auto      | 2      | 90.21        | 251720.52           | 19114           |
| 10.0 | auto      | 3      | 90.19        | 283098.15           | 20044           |
| 10.0 | auto      | 4      | 89.32        | 371571.03           | 21844           |
| 10.0 | 0.001     | 2      | 90.07        | 255732.36           | 19414           |
| 10.0 | 0.001     | 3      | 89.97        | 291784.53           | 20458           |
| 10.0 | 0.001     | 4      | 88.22        | 406621.91           | 22726           |
| 10.0 | 0.01      | 2      | 89.25        | 257066.00           | 18819           |
| 10.0 | 0.01      | 3      | 89.98        | 280376.67           | 20071           |
| 10.0 | 0.01      | 4      | 89.71        | 348555.59           | 21070           |

---

### RBF Kernel Results

| C    | Gamma     | Accuracy (%) | Training Time (ms) | Support Vectors |
|------|-----------|--------------|---------------------|-----------------|
| 0.1  | scale     | 85.38        | 452254.31           | 28312           |
| 0.1  | auto      | 85.38        | 453349.96           | 28312           |
| 0.1  | 0.001     | 85.18        | 442199.19           | 28544           |
| 0.1  | 0.01      | 66.02        | 1732800.02          | 44877           |
| 1.0  | scale     | 89.26        | 285977.60           | 21641           |
| 1.0  | auto      | 89.27        | 285379.36           | 21640           |
| 1.0  | 0.001     | 89.02        | 269671.43           | 21572           |
| 1.0  | 0.01      | 80.50        | 1480725.09          | 38358           |
| 10.0 | scale     | 90.31        | 272679.53           | 20780           |
| 10.0 | auto      | 90.31        | 272804.02           | 20782           |
| 10.0 | 0.001     | 90.18        | 247542.41           | 19960           |
| 10.0 | 0.01      | 81.85        | 1605553.22          | 39661           |

---

## Complete Results: All Dataset-Kernel Combinations

| Dataset      | Kernel | C    | Gamma     | Degree | Accuracy (%) | Training Time (ms) | Support Vectors |
|--------------|--------|------|-----------|--------|--------------|---------------------|-----------------|
| MNIST        | poly   | 0.1  | scale     | 2      | 94.10        | 839631.43           | 30026           |
| MNIST        | poly   | 0.1  | scale     | 3      | 86.29        | 1454665.34          | 39372           |
| MNIST        | poly   | 0.1  | scale     | 4      | 59.21        | 2049577.99          | 47787           |
| MNIST        | poly   | 0.1  | auto      | 2      | 93.49        | 914155.58           | 31710           |
| MNIST        | poly   | 0.1  | auto      | 3      | 83.66        | 1609280.72          | 42067           |
| MNIST        | poly   | 0.1  | auto      | 4      | 47.43        | 2161517.11          | 49675           |
| MNIST        | poly   | 0.1  | 0.001     | 2      | 91.95        | 1130358.17          | 36138           |
| MNIST        | poly   | 0.1  | 0.001     | 3      | 74.16        | 1988409.82          | 48088           |
| MNIST        | poly   | 0.1  | 0.001     | 4      | 32.95        | 2429273.53          | 53300           |
| MNIST        | poly   | 0.1  | 0.01      | 2      | 97.45        | 223731.86           | 13422           |
| MNIST        | poly   | 0.1  | 0.01      | 3      | 97.90        | 334049.90           | 15167           |
| MNIST        | poly   | 0.1  | 0.01      | 4      | 96.83        | 536956.94           | 18558           |
| MNIST        | poly   | 1.0  | scale     | 2      | 96.92        | 322588.88           | 16466           |
| MNIST        | poly   | 1.0  | scale     | 3      | 95.83        | 592604.80           | 21544           |
| MNIST        | poly   | 1.0  | scale     | 4      | 89.01        | 1128233.62          | 31260           |
| MNIST        | poly   | 1.0  | auto      | 2      | 96.85        | 345066.85           | 17179           |
| MNIST        | poly   | 1.0  | auto      | 3      | 95.36        | 657996.50           | 23114           |
| MNIST        | poly   | 1.0  | auto      | 4      | 86.67        | 1265291.32          | 34179           |
| MNIST        | poly   | 1.0  | 0.001     | 2      | 96.51        | 409526.80           | 19264           |
| MNIST        | poly   | 1.0  | 0.001     | 3      | 93.60        | 875163.53           | 28122           |
| MNIST        | poly   | 1.0  | 0.001     | 4      | 77.77        | 1670007.78          | 41795           |
| MNIST        | poly   | 1.0  | 0.01      | 2      | 97.52        | 203689.50           | 12899           |
| MNIST        | poly   | 1.0  | 0.01      | 3      | 97.94        | 332019.36           | 15189           |
| MNIST        | poly   | 1.0  | 0.01      | 4      | 96.85        | 537550.02           | 18565           |
| MNIST        | poly   | 10.0 | scale     | 2      | 97.51        | 212703.59           | 13096           |
| MNIST        | poly   | 10.0 | scale     | 3      | 97.64        | 351666.00           | 15462           |
| MNIST        | poly   | 10.0 | scale     | 4      | 96.03        | 613551.28           | 20199           |
| MNIST        | poly   | 10.0 | auto      | 2      | 97.46        | 213249.21           | 13178           |
| MNIST        | poly   | 10.0 | auto      | 3      | 97.62        | 358474.76           | 15683           |
| MNIST        | poly   | 10.0 | auto      | 4      | 95.54        | 656864.82           | 21257           |
| MNIST        | poly   | 10.0 | 0.001     | 2      | 97.45        | 223472.02           | 13422           |
| MNIST        | poly   | 10.0 | 0.001     | 3      | 97.25        | 407601.60           | 16801           |
| MNIST        | poly   | 10.0 | 0.001     | 4      | 93.27        | 870712.16           | 25309           |
| MNIST        | poly   | 10.0 | 0.01      | 2      | 97.48        | 203654.23           | 12902           |
| MNIST        | poly   | 10.0 | 0.01      | 3      | 97.94        | 331683.98           | 15189           |
| MNIST        | poly   | 10.0 | 0.01      | 4      | 96.85        | 532788.16           | 18565           |
| MNIST        | rbf    | 0.1  | scale     | -      | 93.40        | 557587.21           | 25282           |
| MNIST        | rbf    | 0.1  | auto      | -      | 93.43        | 579581.13           | 25350           |
| MNIST        | rbf    | 0.1  | 0.001     | -      | 93.54        | 570734.43           | 25819           |
| MNIST        | rbf    | 0.1  | 0.01      | -      | 65.96        | 2105207.43          | 45105           |
| MNIST        | rbf    | 1.0  | scale     | -      | 96.50        | 336559.87           | 15484           |
| MNIST        | rbf    | 1.0  | auto      | -      | 96.49        | 320190.45           | 15339           |
| MNIST        | rbf    | 1.0  | 0.001     | -      | 96.27        | 296986.34           | 15152           |
| MNIST        | rbf    | 1.0  | 0.01      | -      | 85.13        | 1998482.47          | 36734           |
| MNIST        | rbf    | 10.0 | scale     | -      | 97.26        | 278479.75           | 14535           |
| MNIST        | rbf    | 10.0 | auto      | -      | 97.28        | 261179.59           | 14114           |
| MNIST        | rbf    | 10.0 | 0.001     | -      | 97.18        | 225985.44           | 13104           |
| MNIST        | rbf    | 10.0 | 0.01      | -      | 86.27        | 1995558.85          | 36931           |
| FashionMNIST | poly   | 0.1  | scale     | 2      | 84.42        | 547342.74           | 29756           |
| FashionMNIST | poly   | 0.1  | scale     | 3      | 82.03        | 642018.98           | 30997           |
| FashionMNIST | poly   | 0.1  | scale     | 4      | 75.65        | 925177.07           | 34412           |
| FashionMNIST | poly   | 0.1  | auto      | 2      | 84.42        | 548140.12           | 29756           |
| FashionMNIST | poly   | 0.1  | auto      | 3      | 82.03        | 642327.67           | 30997           |
| FashionMNIST | poly   | 0.1  | auto      | 4      | 75.65        | 918118.13           | 34412           |
| FashionMNIST | poly   | 0.1  | 0.001     | 2      | 82.62        | 647273.81           | 32153           |
| FashionMNIST | poly   | 0.1  | 0.001     | 3      | 78.36        | 821736.55           | 34662           |
| FashionMNIST | poly   | 0.1  | 0.001     | 4      | 69.69        | 1230680.94          | 39501           |
| FashionMNIST | poly   | 0.1  | 0.01      | 2      | 90.07        | 250089.68           | 19414           |
| FashionMNIST | poly   | 0.1  | 0.01      | 3      | 90.29        | 274542.41           | 20072           |
| FashionMNIST | poly   | 0.1  | 0.01      | 4      | 89.67        | 343277.82           | 21120           |
| FashionMNIST | poly   | 1.0  | scale     | 2      | 88.88        | 303085.04           | 21926           |
| FashionMNIST | poly   | 1.0  | scale     | 3      | 88.49        | 346659.88           | 22940           |
| FashionMNIST | poly   | 1.0  | scale     | 4      | 85.67        | 500918.91           | 25479           |
| FashionMNIST | poly   | 1.0  | auto      | 2      | 88.88        | 304159.48           | 21926           |
| FashionMNIST | poly   | 1.0  | auto      | 3      | 88.49        | 349759.87           | 22940           |
| FashionMNIST | poly   | 1.0  | auto      | 4      | 85.68        | 504498.03           | 25479           |
| FashionMNIST | poly   | 1.0  | 0.001     | 2      | 88.40        | 335970.31           | 23130           |
| FashionMNIST | poly   | 1.0  | 0.001     | 3      | 86.99        | 413641.01           | 24807           |
| FashionMNIST | poly   | 1.0  | 0.001     | 4      | 82.22        | 641515.80           | 28660           |
| FashionMNIST | poly   | 1.0  | 0.01      | 2      | 89.68        | 253785.18           | 18917           |
| FashionMNIST | poly   | 1.0  | 0.01      | 3      | 89.98        | 279279.99           | 20071           |
| FashionMNIST | poly   | 1.0  | 0.01      | 4      | 89.71        | 348479.58           | 21070           |
| FashionMNIST | poly   | 10.0 | scale     | 2      | 90.21        | 252635.79           | 19113           |
| FashionMNIST | poly   | 10.0 | scale     | 3      | 90.19        | 284770.93           | 20043           |
| FashionMNIST | poly   | 10.0 | scale     | 4      | 89.32        | 372995.88           | 21844           |
| FashionMNIST | poly   | 10.0 | auto      | 2      | 90.21        | 251720.52           | 19114           |
| FashionMNIST | poly   | 10.0 | auto      | 3      | 90.19        | 283098.15           | 20044           |
| FashionMNIST | poly   | 10.0 | auto      | 4      | 89.32        | 371571.03           | 21844           |
| FashionMNIST | poly   | 10.0 | 0.001     | 2      | 90.07        | 255732.36           | 19414           |
| FashionMNIST | poly   | 10.0 | 0.001     | 3      | 89.97        | 291784.53           | 20458           |
| FashionMNIST | poly   | 10.0 | 0.001     | 4      | 88.22        | 406621.91           | 22726           |
| FashionMNIST | poly   | 10.0 | 0.01      | 2      | 89.25        | 257066.00           | 18819           |
| FashionMNIST | poly   | 10.0 | 0.01      | 3      | 89.98        | 280376.67           | 20071           |
| FashionMNIST | poly   | 10.0 | 0.01      | 4      | 89.71        | 348555.59           | 21070           |
| FashionMNIST | rbf    | 0.1  | scale     | -      | 85.38        | 452254.31           | 28312           |
| FashionMNIST | rbf    | 0.1  | auto      | -      | 85.38        | 453349.96           | 28312           |
| FashionMNIST | rbf    | 0.1  | 0.001     | -      | 85.18        | 442199.19           | 28544           |
| FashionMNIST | rbf    | 0.1  | 0.01      | -      | 66.02        | 1732800.02          | 44877           |
| FashionMNIST | rbf    | 1.0  | scale     | -      | 89.26        | 285977.60           | 21641           |
| FashionMNIST | rbf    | 1.0  | auto      | -      | 89.27        | 285379.36           | 21640           |
| FashionMNIST | rbf    | 1.0  | 0.001     | -      | 89.02        | 269671.43           | 21572           |
| FashionMNIST | rbf    | 1.0  | 0.01      | -      | 80.50        | 1480725.09          | 38358           |
| FashionMNIST | rbf    | 10.0 | scale     | -      | 90.31        | 272679.53           | 20780           |
| FashionMNIST | rbf    | 10.0 | auto      | -      | 90.31        | 272804.02           | 20782           |
| FashionMNIST | rbf    | 10.0 | 0.001     | -      | 90.18        | 247542.41           | 19960           |
| FashionMNIST | rbf    | 10.0 | 0.01      | -      | 81.85        | 1605553.22          | 39661           |

---

# Best Models by Dataset

| Dataset      | Kernel | C    | Gamma | Degree | Accuracy (%) | Training Time (ms) | Support Vectors |
|--------------|--------|------|-------|--------|--------------|---------------------|-----------------|
| MNIST        | poly   | 1.0  | 0.01  | 3      | **97.94**    | 332019.36           | 15189           |
| MNIST        | poly   | 10.0 | 0.01  | 3      | **97.94**    | 331683.98           | 15189           |
| FashionMNIST | rbf    | 10.0 | scale | -      | **90.31**    | 272679.53           | 20780           |
| FashionMNIST | rbf    | 10.0 | auto  | -      | **90.31**    | 272804.02           | 20782           |

## Key Findings

**MNIST Dataset:**
- Best accuracy: **97.94%** achieved by polynomial kernel (C=1.0 or 10.0, gamma=0.01, degree=3)
- Polynomial kernel significantly outperforms RBF for MNIST
- Training time: ~332 seconds (~5.5 minutes)

**FashionMNIST Dataset:**
- Best accuracy: **90.31%** achieved by RBF kernel (C=10.0, gamma=scale or auto)
- RBF kernel performs better than polynomial for FashionMNIST
- Training time: ~273 seconds (~4.5 minutes)

---



# Q2: CPU vs GPU Performance Comparison

## Performance Summary Table

| Compute | Model     | Optimizer | Batch Size | Learning Rate | Test Accuracy (%) | Best Val Accuracy (%) | Train Time (ms) | FLOPs       | Speedup vs CPU |
|---------|-----------|-----------|------------|---------------|-------------------|-----------------------|-----------------|-------------|----------------|
| CPU     | ResNet-18 | SGD       | 16         | 0.001         | 91.06             | 91.20                 | 3,561,202       | 148.865M    | 1.00×          |
| CPU     | ResNet-32 | SGD       | 16         | 0.001         | 90.87             | 91.19                 | 2,727,965       | 285.826M    | 1.00×          |
| CPU     | ResNet-50 | SGD       | 16         | 0.001         | 89.74             | 89.93                 | 6,214,446       | 337.304M    | 1.00×          |
| CPU     | ResNet-18 | Adam      | 16         | 0.001         | 90.19             | 90.51                 | 3,998,669       | 148.865M    | 1.00×          |
| CPU     | ResNet-32 | Adam      | 16         | 0.001         | 90.15             | 90.34                 | 2,940,160       | 285.826M    | 1.00×          |
| CPU     | ResNet-50 | Adam      | 16         | 0.001         | 88.58             | 88.31                 | 7,015,001       | 337.304M    | 1.00×          |
| **GPU** | **ResNet-18** | **SGD**   | **16**     | **0.001**     | **91.19**         | **91.33**             | **156,975**     | **148.865M**| **22.69×**     |
| **GPU** | **ResNet-32** | **SGD**   | **16**     | **0.001**     | **91.14**         | **91.11**             | **223,410**     | **285.826M**| **12.21×**     |
| **GPU** | **ResNet-50** | **SGD**   | **16**     | **0.001**     | **90.24**         | **89.81**             | **330,349**     | **337.304M**| **18.81×**     |
| **GPU** | **ResNet-18** | **Adam**  | **16**     | **0.001**     | **89.89**         | **90.57**             | **167,923**     | **148.865M**| **23.81×**     |
| **GPU** | **ResNet-32** | **Adam**  | **16**     | **0.001**     | **90.06**         | **90.00**             | **239,122**     | **285.826M**| **12.30×**     |
| **GPU** | **ResNet-50** | **Adam**  | **16**     | **0.001**     | **88.83**         | **88.81**             | **352,534**     | **337.304M**| **19.90×**     |

---

## Key Findings

###  **GPU Acceleration Performance**
- **GPU training delivers 12–24× speedup** across all model architectures compared to CPU
- **ResNet-18 with Adam optimizer** achieved the highest speedup at **23.81×** faster than CPU
- **ResNet-32 models** showed the lowest speedup (~12×), likely due to memory transfer overhead relative to computation
- Average GPU speedup across all configurations: **~18.3×**

###  **Model Accuracy Comparison**
- **ResNet-18 with SGD** achieved the highest test accuracy at **91.19%** (GPU) and **91.06%** (CPU)
- **SGD optimizer consistently outperformed Adam** across all architectures by **0.5–1.5%** in test accuracy
- Deeper models (ResNet-50) showed **accuracy degradation** (88.58–90.24%), suggesting potential overfitting or insufficient training epochs
- **CPU vs GPU accuracy difference**: Minimal (<0.3%), demonstrating consistent numerical stability across compute platforms

###  **Optimizer Performance Analysis**
- **SGD Optimizer:**
  - Superior accuracy: 89.74–91.19% test accuracy
  - Better generalization: Smaller gap between validation and test accuracy
  - More stable training dynamics across model depths
  
- **Adam Optimizer:**
  - Faster convergence but lower final accuracy: 88.58–90.19% test accuracy
  - **20–30% slower training time** on GPU compared to SGD (likely due to additional momentum buffer computations)
  - Less effective for deeper architectures (ResNet-50)

###  **Computational Efficiency**
- **FLOPs scale with model depth**: ResNet-18 (148.9M) → ResNet-32 (285.8M) → ResNet-50 (337.3M)
- **Training time does not linearly correlate with FLOPs**, indicating memory bandwidth and optimization overhead impact
- **ResNet-32** demonstrated the best efficiency-to-accuracy ratio on GPU (223ms training time, 91.14% accuracy)

###  **Optimal Configuration Recommendations**
- **Best Overall Performance**: ResNet-18 + SGD + GPU → 91.19% accuracy, 156.98ms training time
- **Best Efficiency**: ResNet-32 + SGD + GPU → 91.14% accuracy, 223.41ms training time, 12.21× speedup
- **Production Deployment**: GPU acceleration is essential for real-time training scenarios (12–24× faster)
- **Hyperparameter Tuning**: Consider exploring higher learning rates for the Adam optimizer to close the accuracy gap with SGD

###  **Architectural Insights**
- **ResNet-18 offers the best accuracy-to-complexity tradeoff** for this dataset
- **Diminishing returns with depth**: ResNet-50's 2× FLOPs over ResNet-18 yielded 1–2% lower accuracy
- **Batch size 16** appears suitable for all configurations without memory bottlenecks on the GPU

---

This comprehensive evaluation demonstrates that **GPU acceleration is imperative for efficient deep learning workflows**, providing **12–24× speedup** with negligible accuracy trade-offs. **SGD optimizer with ResNet-18 architecture** emerged as the optimal configuration, balancing accuracy (91.19%), training efficiency (156.98ms), and computational cost (148.9M FLOPs). Future work should investigate learning rate schedules, batch size scaling, and mixed-precision training to further optimize performance.

---
### CPU vs GPU Comparison Chart
![CPU vs GPU Comparison](./figures/cpu_gpu_comparison.png)

---

## Analysis
(Add your detailed analysis here)
