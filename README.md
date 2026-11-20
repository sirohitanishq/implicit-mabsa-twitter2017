# ImplicitMABSA: Multimodal Implicit Aspect-Based Sentiment Analysis Using Text and Images

Compares text-only, image-only and multimodal models for target-level sentiment
(negative / neutral / positive) on **Twitter-2017** (tweet text + image), with the
aspect term shown, masked or removed, to study how much visual context helps when
the aspect is not explicitly available in the text.

## Models

| Name | Description |
|---|---|
| `text` | BERT-base-uncased, [CLS] representation, linear classifier |
| `image` | ResNet-50 (ImageNet pretrained), global-pooled features, linear classifier |
| `concat` | BERT [CLS] + ResNet-50 pooled features, concatenated, MLP classifier |
| `cross` | BERT token states attend to ResNet-50 region features (multi-head cross-attention), residual + LayerNorm, classifier on [CLS] |

## Aspect modes

| Mode | Input to the text encoder |
|---|---|
| `explicit` | `sentence` paired with `aspect` as a second segment: `[CLS] sentence [SEP] aspect [SEP]` |
| `mask` | aspect term replaced by `[MASK]` (position known, identity hidden) |
| `remove` | aspect term deleted from the sentence (position and identity hidden) |

## Setup

```
pip install torch torchvision transformers pillow scikit-learn numpy
```

Tested with Python 3.x, PyTorch 2.6 (CUDA 12.4) on Windows. A CUDA GPU is strongly recommended.

## Data

1. Clone the TomBERT repo into `Twitter_17/`:
   ```
   git clone https://github.com/jefferyYu/TomBERT Twitter_17/TomBERT
   ```
   The files used are `absa_data/twitter/{train,dev,test}.tsv`
   (3562 / 1176 / 1234 samples).
2. Download the tweet images from the link in the TomBERT README
   (Google Drive) and extract them so that the Twitter-2017 images are in:
   ```
   data/IJCAI2019_data/IJCAI2019_data/twitter2017_images
   ```
3. If your folders differ, edit `TEXT_DIR` and `IMG_DIR` at the top of `dataset.py`.

The dataset and images are not included in this repository.

## Usage

```
python dataset.py                                   # sanity check: counts, missing images, sample
python train.py --model all --aspect mask           # train all 4 models, masked aspect
python train.py --model cross --aspect remove --seeds 1 2 3
python aggregate.py                                 # mean ± std table from the results files
```

Training options: `--model {text,image,concat,cross,all}`,
`--aspect {explicit,mask,remove}`, `--epochs 5`, `--batch_size 16`,
`--workers 4` (use `--workers 0` if DataLoader errors on Windows), `--seeds 42 1 2 ...`.

Each run keeps the epoch with the best dev macro-F1 and reports test accuracy and
macro precision / recall / F1. Results are appended to `results.csv`
(`results_seed42.csv` holds the first seed-42 masked runs, which `aggregate.py` also reads).

Training setup: AdamW, lr 2e-5 for BERT/ResNet and 1e-4 for new layers, 5 epochs,
batch size 16, linear warmup/decay, mixed precision, max sequence length 64.

## Results (test set, %, macro-averaged)

| Aspect | Model | n | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| mask | Text | 4 | 66.00 ± 0.61 | 64.96 ± 0.99 | 61.83 ± 1.11 | 63.02 ± 1.04 |
| mask | Image | 1 | 47.41 | 31.57 | 36.51 | 33.86 |
| mask | Concat | 4 | 67.10 ± 0.72 | 65.28 ± 0.57 | 64.12 ± 2.18 | 64.55 ± 1.40 |
| mask | Cross-attention | 4 | **67.36 ± 0.79** | **66.20 ± 1.08** | 64.08 ± 1.11 | **64.94 ± 0.66** |
| explicit | Text | 1 | 67.34 | 67.59 | 62.24 | 63.74 |
| explicit | Concat | 1 | 68.07 | 66.24 | 65.66 | 65.82 |
| explicit | Cross-attention | 1 | 66.29 | 64.66 | 64.42 | 64.02 |
| remove | Text | 1 | 63.45 | 62.29 | 58.09 | 59.58 |
| remove | Concat | 1 | 64.02 | 62.05 | 59.23 | 60.22 |
| remove | Cross-attention | 1 | 62.64 | 61.01 | 57.50 | 58.55 |

`n` is the number of random seeds. Rows with `n = 1` are single runs and are only indicative.

**Findings**
- Image-only is close to chance, so the image alone cannot identify the sentiment toward a specific aspect.
- With the aspect masked, adding the image gives a modest gain of about 1.5 to 1.9 F1 over text-only, and cross-attention is the most stable fusion (lowest variance).
- Removing the aspect entirely lowers all models by about 4 F1 points; the benefit of the image is small and not consistent in that single-seed setting.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Data loading, image indexing, aspect explicit / mask / remove |
| `train.py` | Models, training loop, evaluation, results logging |
| `aggregate.py` | Mean ± std table over seeds |
| `results.csv`, `results_seed42.csv` | Raw per-run test results |

## Reference

Yu and Jiang, "Adapting BERT for Target-Oriented Multimodal Sentiment Classification", IJCAI 2019 (Twitter-2015 / Twitter-2017 dataset and splits).
