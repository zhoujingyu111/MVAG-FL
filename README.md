MVAG-FL: Multi-View Attention Guided Feature Learning for Industrial Anomaly Detection

The code is a preliminary version, so it may be somewhat messy with minor errors. We plan to refine and reformat it after the paper is accepted.

Dataset：
Real-IAD： A new large-scale challenging industrial AD dataset, containing 30 classes with totally 151,050 images; 2,000 ∼ 5,000 resolution; 0.01% ~ 6.75% defect proportions; 1:1 ~ 1:10 defect ratio.
Download and extract Real-IAD into data/realiad.  ：https://realiad4ad.github.io/Real-IAD/


Environments:
Create a new conda environment and install required packages.

conda create -n env python=3.8.12
conda activate env
pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard fvcore accimage Ninja
pip3 install matplotlib==3.2.1   numpy==1.18.4   opencv_python_headless==4.6.0.66   pandas==1.3.5   Pillow==9.0.1   scikit_image==0.19.3   scikit_learn==0.22.2.post1
pip3 install scipy==1.4.1
pip3 install tabulate==0.9.0
pip3 install torch==1.12.0+cu113
pip3 install torchvision==0.13.0+cu113
pip3 install tqdm==4.64.1
pip3 install ptflops==0.7
pip3 install timm==0.9.12
pip3 install mmdet==2.25.3
pip3 install --upgrade protobuf==3.20.1 scikit-image faiss-gpu
pip3 install adeval
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install fastprogress geomloss FrEIA mamba_ssm adeval fvcore==0.1.5.post20221221
(or) conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia


All experiments are trained for 50,000 iterations on a single NVIDIA RTX A6000 with the Pytorch framework.


Evaluation Metrics： Both image-level and pixel-level met-rics are typically used to evaluate algorithm performance in USDD. Image-level metrics assess whether an entire product is anomalous, while pixel-level metrics measure defect localization and can further evaluate defect severity. Based on previous defect detection work, seven evaluation metrics are employed. Image-level performance is evaluated using Area Under the Receiver Operator Curve (AUROC), Average Precision (AP), and F1 score at the optimal threshold (F-max). Pixel-level performance is measured by AUROC, AP, F-max, and the Area Under the Per Region Overlap (AUPRO). To provide a comprehensive assessment of the model’s perfor-mance, we compute the average of the all-evaluation metrics mentioned above, referred to as mAD. The final dataset result is calculated as the average across all classes.


Results:
![image](https://github.com/user-attachments/assets/3c137abd-628e-42a3-a609-e779d7092476)
