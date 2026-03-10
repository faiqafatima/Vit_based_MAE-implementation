# Vit_based_MAE-implementation
Masked Autoencoders (MAE) for Image Reconstruction
This repository contains the implementation of a Self-Supervised Image Representation Learning system using Masked Autoencoders (MAE), trained on the TinyImageNet dataset.

Features
Asymmetric Architecture: ViT-Base Encoder and Lightweight Decoder.

Masking: 75% random patch masking during training.

Optimization: AdamW with Cosine Annealing Scheduler and Mixed Precision (FP16).

Results: Achieved an average PSNR of 20.55 dB and SSIM of 0.4772.
Model Weights
Due to GitHub's file size limits, the trained model weights (best_model.pth) are hosted here: [https://www.kaggle.com/code/faiqarana/ai-ass02-22f-3278/edit/run/302434876]
## Qualitative Results
Below are the reconstructions from the test set (TinyImageNet). The model effectively recovers structure and color even with 75% of pixels missing.

## Quantitative Performance
- **Average PSNR:** 20.55 dB
- **Average SSIM:** 0.4772
