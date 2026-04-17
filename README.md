MVAG-FL: Multi-View Attention Guided Feature Learning for Industrial Anomaly Detection
License
Python 3.8+
PyTorch 2.1+

⚠️ Note: This is a preliminary version of the code. It may contain minor errors and requires refinement. We plan to clean and reformat the code after paper acceptance.

📖 Introduction
MVAG-FL is a novel anomaly detection framework designed for industrial inspection scenarios. The method leverages multi-view attention guidance to enhance feature learning, achieving state-of-the-art performance on challenging industrial anomaly detection benchmarks.

✨ Key Features
Multi-View Attention: Guides feature learning from multiple perspectives
Industrial-Scale Performance: Tested on high-resolution industrial datasets
Comprehensive Metrics: Evaluated on both image-level and pixel-level metrics
Efficient Training: 50K iterations on single RTX A6000 GPU
📊 Dataset
Real-IAD Dataset
A large-scale challenging industrial anomaly detection dataset:

30 classes with 151,050 images total
High resolution: 2,000 ∼ 5,000 pixels
Low defect proportions: 0.01% ~ 6.75%
Imbalanced defect ratio: 1:1 ~ 1:10
Setup Instructions
Download Real-IAD from https://realiad4ad.github.io/Real-IAD/
Extract to data/realiad/
🛠️ Installation
Environment Setup
Bash
# Create conda environment
conda create -n mvagnn-fl python=3.8.12
conda activate mvagnn-fl

# Install core dependencies
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Or using conda
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional packages
pip install timm==0.9.12 pandas transformers openpyxl imgaug \
            numpy==1.18.4 opencv-python-headless==4.6.0.66 \
            scikit-image==0.19.3 scikit-learn==0.22.2.post1 \
            mmdet==2.25.3 fvcore fastprogress geomloss \
            mamba_ssm adeval faiss-gpu
📈 Evaluation Metrics
We employ comprehensive evaluation metrics following standard anomaly detection protocols:

Image-Level Metrics
AUROC: Area Under the Receiver Operator Curve
AP: Average Precision
F-max: F1 score at optimal threshold
Pixel-Level Metrics
AUROC: Area Under the ROC Curve
AP: Average Precision
F-max: F1 score at optimal threshold
AUPRO: Area Under the Per Region Overlap
Overall Metric
mAD: Average of all evaluation metrics above
📊 Results
Performance on Real-IAD Dataset
Real-IAD Results

Performance on MVTec AD and VisA Datasets
MVTec and VisA Results

🚀 Training
Hardware Requirements
GPU: NVIDIA RTX A6000 (or equivalent with 48GB+ VRAM)
Training Time: ~50,000 iterations
Framework: PyTorch
Quick Start
Bash
# Example training command (subject to code structure)
python train.py --dataset realiad --config configs/mvagnn_fl.yaml
📝 Citation
If you use this code in your research, please cite our paper (to be updated upon acceptance):

 @article{mvagnnfl2024,
  title={MVAG-FL: Multi-View Attention Guided Feature Learning for Industrial Anomaly Detection},
  author={Anonymous Authors},
  journal={Under Review},
  year={2024}
}
🤝 Contributing
We welcome contributions! Please feel free to submit a Pull Request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📧 Contact
For questions or discussions, please open an issue in the GitHub repository.

Note: Code will be refined and properly formatted after paper acceptance.
