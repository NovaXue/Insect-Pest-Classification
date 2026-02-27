# Insect-Pest-Classification
# Insect Pest Species Identification (IP102) — Deep Learning & Dual-Stream Fusion

## Overview
Insect pest species identification is critical for agriculture, pest control, and ecological monitoring. Fast and accurate recognition enables early detection of invasive species, supports timely intervention, and helps preserve crop yield and quality. However, manual identification based on visual inspection is time-consuming, subjective, and unreliable in real-world conditions (e.g., cluttered backgrounds, varying lighting, small or partially occluded insects).

This project builds an automated insect pest recognition system using deep learning, aiming for strong classification performance and robustness under challenging “in-the-wild” image conditions.

---

## Dataset
We use **IP102**, a benchmark dataset for insect pest recognition containing **75,222 images across 102 pest species**. Images exhibit large variation in resolution, viewpoint, lighting, insect scale, and background complexity, making the task realistic and challenging.

Dataset repository:  
https://github.com/xpwu95/IP102/tree/master

> Note: The dataset is not included in this repository due to its size. Please download it from the official source above.

---

## Problem & Challenges
The task is to classify an input image into one of **102 insect pest species**. Key difficulties include:

- **High inter-class similarity** — different species can look visually similar  
- **High intra-class variability** — the same species appears in different poses, lighting, or occlusion  
- **Complex backgrounds** — natural scenes with clutter  
- **Class imbalance** — some species have significantly more samples than others  

These factors make robust recognition substantially more difficult than clean laboratory datasets.

---

## Methods

### Baseline Model Benchmarking (CNN)
We implemented and compared several widely used pretrained CNN backbones:

- ResNet50  
- DenseNet121  
- EfficientNet-B0  
- MobileNetV2  

These baselines provide a strong performance reference and allow comparison between accuracy and efficiency trade-offs.

---

### Proposed Method: Dual-Stream Fusion (Transformer + Transformer)

To improve recognition under fine-grained similarity and real-world variability, we designed a **custom dual-stream fusion architecture** integrating:

- **Swin Transformer**
- **BEiT**

Each stream extracts high-level visual representations independently. The features are then fused to form a more discriminative joint representation before final classification.

This fusion strategy leverages complementary strengths of different vision transformers and achieved improved classification performance compared to the CNN baselines.

---

## Preprocessing & Training
Typical preprocessing steps include:

- Image resizing to a consistent input size  
- Data augmentation (random crop, flip, color jitter)  
- Normalization and tensor conversion  

Transfer learning was applied using pretrained weights, improving convergence and generalization.

---

## Results Summary
We report comparative results across:

- Multiple pretrained CNN architectures  
- Our dual-stream Swin + BEiT fusion model  

The proposed fusion model achieved improved performance over standard baselines, particularly in challenging categories with high visual similarity and environmental variability.

(See the notebook and report for detailed experimental results.)

---

## Repository Contents
- `COMP9444.ipynb` — training and evaluation notebook  
- `report.pdf` — project report (methodology, comparisons, analysis)

---

## Applications
This system supports real-world deployment scenarios such as:

- Smart agriculture pest monitoring  
- Pest early warning systems  
- Crop yield protection  
- Ecological surveillance  

---

## Technologies
- Python  
- PyTorch  
- timm  
- Jupyter Notebook

---

## Relevant Papers

**[1]** X. Wu, C. Zhan, Y.-K. Lai, M.-M. Cheng, and J. Yang,  
“IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition,”  
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.  
DOI: https://doi.org/10.1109/CVPR.2019.00899  
IEEE: https://ieeexplore.ieee.org/document/8954351  
Open Access PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_IP102_A_Large_Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.pdf  

---

**[2]** A. Setiawan, N. Yudistira, and R. C. Wihandika,  
“Large scale pest classification using efficient Convolutional Neural Network with augmentation and regularizers,”  
*Computers and Electronics in Agriculture*, Vol. 200, 2022.  
https://www.sciencedirect.com/science/article/pii/S0168169922005191  

---

**[3]** W. Linfeng, L. Yong, L. Jiayao, W. Yunsheng, and X. Shipu,  
“Based on the multi-scale information sharing network of fine-grained attention for agricultural pest detection,”  
*PLOS ONE*, 18(10): e0286732.  
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286732  

---

**[4]** J. An, Y. Du, P. Hong, L. Zhang, and X. Weng,  
“Insect recognition based on complementary features from multiple views,”  
*Scientific Reports*, 13(1):2966, 2023.  
DOI: https://doi.org/10.1038/s41598-023-29600-1  
https://europepmc.org/article/pmc/pmc9940688  

---

**[5]** S. Kar et al.,  
“Self-supervised learning improves classification of agriculturally important insect pests in plants,”  
*The Plant Phenome Journal*, 6, e20079.  
https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/ppj2.20079  
