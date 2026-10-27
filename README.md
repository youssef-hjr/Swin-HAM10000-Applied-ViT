# Fine-Tuning a Swin Transformer for Skin Lesion Classification on HAM10000

**Youssef Hajri, Student ID 1425158**
MAAI901 Deep Learning for Visual Data, Assessment 2 (Applied Vision Transformer Report), University of Wollongong in Dubai

This repository contains the code for my report *"What Headline Accuracy Hides: Fine-Tuning a Swin Transformer for Skin Lesion Classification on HAM10000"*. I fine-tune Swin-T on the seven HAM10000 classes using a lesion-level split. I compare ImageNet-22k and ImageNet-1k pretraining, and I analyse performance by skin tone and by how each diagnosis was confirmed.

## Files

| File | Description |
|---|---|
| `Swin_HAM10000_1425158.ipynb` | The full notebook, executed with all outputs: data, pre-processing, model, training, evaluation, bias analysis and figures |
| `appendix_figs.py` | Builds the appendix figures from the saved test predictions: `python appendix_figs.py .` after the notebook has run. Run it from the folder that contains `data/` and `run/outputs/`. |
| `outputs/` | Every figure used in the report, plus `results.json` with all the numbers |
| `requirements.txt` | Python packages |

## Data and weights

Place the following files in a folder called `data/` next to the notebook.

1. The ISIC 2018 Task 3 training images and ground truth (this is HAM10000):
   - https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
   - https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip
   - Unzip both inside `data/`.
2. The lesion groupings file, used for the lesion-level split. Save it as `data/groupings.csv`:
   - https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv
3. The official Swin-T checkpoints from the Swin Transformer authors:
   - ImageNet-1k: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth (save it as `data/swin_1k.pth`)
   - ImageNet-22k: https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth (save it as `data/swin_22k.pth`)

## Running

```bash
pip install -r requirements.txt
jupyter notebook Swin_HAM10000_1425158.ipynb
```

On the first run, the notebook resizes all images to 224x224, computes the skin-tone (ITA) proxy and caches both to `data/`. The split and training use a fixed seed (42).

## Main results (test set, 1,500 images)

| Model | Accuracy | Balanced accuracy | Malignant recall | AUC (malignant vs benign) |
|---|---|---|---|---|
| Swin-T 22k, linear probe | 76.6% | 67.3% | 73.9% | 0.904 |
| **Swin-T 22k, fine-tuned** | 68.7% | **68.5%** | **78.4%** | **0.910** |
| Swin-T 1k, linear probe | 74.5% | 64.7% | 66.7% | 0.879 |
| Swin-T 1k, fine-tuned | 68.3% | 67.5% | 71.8% | 0.886 |

The fine-tuned model has lower accuracy than the frozen baseline, yet it detects 13 more of the 291 malignant lesions. The test set contains only 7 malignant lesions on darker skin, which is too few to assess fairness.
