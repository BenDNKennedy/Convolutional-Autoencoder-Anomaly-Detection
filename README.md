# Image-Level Anomaly Detection with a Convolutional Autoencoder

An unsupervised anomaly detection system that learns what "normal" looks like from defect-free images only, then flags defective parts by how badly it fails to reconstruct them. Built in PyTorch for ECE 471 (Machine Vision) at the University of Victoria.

**Top AUROC: 0.86 (screws) / 0.80 (pasta)** — trained with no labelled anomalies, no data augmentation, and no regularization.

<!-- Add a figure here once you upload one — the original/reconstructed/error-map triptych is the single most convincing image in the whole project:
![Original, reconstruction, and error heatmap for a defective screw](figures/error_map_screw.png)
-->

## The idea

Supervised defect detection needs lots of examples of things going wrong, and in a real factory those are exactly the images you don't have. A convolutional autoencoder sidesteps this: train it only on good parts until it reconstructs them accurately, and it never learns how to represent a defect. Feed it a defective part and the reconstruction error spikes — most sharply right where the defect is.

Pipeline:

```
Input image (128×128 RGB)
  → resize + normalize to [0,1]
  → convolutional encoder
  → latent representation
  → decoder
  → reconstructed image
  → per-image MSE  →  anomaly score
```

## Architecture

**Encoder** — three convolutional layers, 3×3 kernel, stride 2, padding 1, channels `3 → 16 → 32 → 256`, ReLU after each. The stride-2 convolutions halve the spatial dimensions at every step, ending at a 16×16×256 latent representation.

**Decoder** — three mirrored transposed convolutions back to `3 × 128 × 128`, ReLU between layers and a final sigmoid to bound outputs to [0,1].

**Loss** — mean squared error between input and reconstruction.

**Training** — Adam, learning rate 0.001, batch size 8, trained on good images only. Optimal epoch counts differed by dataset: pasta peaked around 500 epochs, screws kept improving out to 3000.

**Scoring** — anomaly score is the per-image mean squared reconstruction error. Classification thresholds were swept across all percentiles of the score distribution to find the operating point that maximized F1.

## Results

Peak performance at the best threshold for each dataset:

| Metric | Pasta | Screws |
| --- | --- | --- |
| AUROC | 0.80 | 0.86 |
| Accuracy | 0.8125 | 0.8750 |
| Precision | 0.8889 | 1.0000 |
| Recall | 0.8000 | 0.8000 |
| F1 score | 0.8421 | 0.8889 |
| Threshold percentile | 45 | 50 |

Three findings worth calling out:

**Epochs mattered; batch size and latent dimension barely did.** Sweeping batch size (8, 16) and latent dimension (64, 256) produced little movement in AUROC. Training length dominated everything else.

**More training is not monotonically better.** Past roughly 3000 epochs both datasets fell off sharply — the autoencoder gets good enough to reconstruct defects too, collapsing the error gap the whole method depends on.

**Screws outperformed pasta despite similar object geometry.** The likely reason is image conditions rather than shape: screws have strong edges, metallic highlights, and clean separation from the background, while the pasta images have overlapping pieces, subtle lighting variation, and a uniform matte texture that flattens contrast in the error maps. Background separation and lighting uniformity appear to matter more here than object complexity.

## Repository contents

| File | Description |
| --- | --- |
| `ECE_471_Project.ipynb` | Full implementation — model, training loop, evaluation, and figure generation |
| `ECE471_ProjectReport_Final.pdf` | IEEE-format report with literature review, methodology, and full results |
| `ECE471_PRESENTATION_SLIDES.pdf` | Presentation slides |

## Running it

The notebook is written for Google Colab and downloads the dataset automatically in the first cell. Open it in Colab, run all cells, and it will train and evaluate on both the screws and pasta datasets.

Running locally instead requires `torch`, `torchvision`, `opencv-python`, `numpy`, `matplotlib`, and `scikit-learn`. Note that training is CPU-bound as written; the epoch counts used here are slow without a GPU, and hardware speed turned out to be a real constraint on how much of the parameter space we could explore.

## Why a CAE

Four alternatives were evaluated before settling on the autoencoder:

- **CNN classifiers** — strong when labelled defects are plentiful, which they usually aren't in industrial settings.
- **One-class SVM** — simpler and faster, but weaker AUROC and less robust on complex inputs.
- **k-Nearest Neighbours** — relies on Euclidean distance in feature space and degrades in high dimensions.
- **Local Outlier Factor** — density-based and effective, but shares k-NN's dimensionality limits.

The CAE won on two counts: it is fully unsupervised, and it learns its own compressed feature space rather than depending on distance metrics that break down in high dimensions.

## Limitations and future work

- **No regularization.** L1/L2 penalties or a denoising objective would likely improve generalization and delay the over-training collapse.
- **No data augmentation.** Mirroring and flipping would expand the effective training set.
- **Untuned optimizer.** Adam with a fixed learning rate was used throughout; a learning-rate schedule or an alternative optimizer is unexplored territory.
- **Image-level only.** The error heatmaps localize defects visually, but the system outputs a single score per image rather than a segmentation mask.

The reported numbers reflect an essentially unoptimized baseline, so there is real headroom here.

## Authors

Ben Kennedy · Chase Westlake · Dylan Davis
University of Victoria — ECE 471, Spring 2025

## References

1. B. Staar, M. Lütjen, M. Freitag, "Anomaly detection with convolutional neural networks for industrial surface inspection," *Procedia CIRP*, vol. 79, pp. 484–489, 2019.
2. A. A. Neloy, M. Turgeon, "A comprehensive study of auto-encoders for anomaly detection: efficiency and trade-offs," *Machine Learning with Applications*, vol. 17, 2024.
3. F. Chollet, "Building Autoencoders in Keras," keras.io, 2016.
4. A. Sharma, "Implementing a Convolutional Autoencoder with PyTorch," PyImageSearch, 2023.

Full bibliography in the project report.
