# Scalable Diffusion Models with Transformers (DiT)

A reduced-scale PyTorch implementation of **Diffusion Transformers (DiT)** based on the paper by William Peebles and Saining Xie (ICCV 2023). This project explores Transformer-based diffusion in pretrained VAE latent space and investigates the effects of conditioning mechanisms and model depth on noise-prediction performance.

## Overview

Diffusion Transformers replace the conventional convolutional denoising network with a Transformer operating on patchified latent representations. This project implements a small-scale DiT model trained on CIFAR-10 and evaluates two architectural choices:

- **Conditioning mechanism:** Adaptive Layer Normalization (AdaLN) versus AdaLN-Zero.
- **Model depth:** Four versus six Transformer blocks.

The implementation uses a pretrained Stable Diffusion VAE to encode images into latent representations. The DiT model learns to predict the Gaussian noise added to these latents during the forward diffusion process.

> **Scope:** This is a reduced-scale implementation for studying DiT architecture and ablations. It is not a reproduction of the original paper's full ImageNet benchmarks.

## Reference Paper

Peebles, W., and Xie, S., "Scalable Diffusion Models with Transformers," *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 4195–4205.

- [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)
- [arXiv](https://arxiv.org/abs/2212.09748)

## Methodology

The pipeline consists of the following stages:

1. **VAE encoding:** CIFAR-10 images are encoded into latent representations using a pretrained Stable Diffusion VAE.
2. **Patchification:** Each latent tensor of shape `4 × 8 × 8` is divided into non-overlapping patches of size `2 × 2`, producing 16 tokens.
3. **Conditioning:** Sinusoidal timestep embeddings and class-label embeddings provide the conditioning information.
4. **Transformer denoising:** Transformer blocks process the patch tokens using either AdaLN or AdaLN-Zero conditioning.
5. **Noise prediction:** The final layer reconstructs a latent-shaped tensor representing the predicted noise.
6. **Training objective:** The model is optimized using mean squared error between the predicted and sampled Gaussian noise.
7. **Generation:** Starting from Gaussian noise, a reverse diffusion process produces latent samples that are decoded by the pretrained VAE.

## Experimental Setup

| Configuration                | Value             |
| ---------------------------- | ----------------- |
| Dataset                      | CIFAR-10          |
| Framework                    | PyTorch           |
| Latent shape                 | `4 × 8 × 8`       |
| Patch size                   | `2 × 2`           |
| Transformer hidden dimension | 192               |
| Attention heads              | 3                 |
| Training steps per model     | 100,000           |
| Batch size                   | 128               |
| Learning rate                | `1e-4`            |
| Conditioning variants        | AdaLN, AdaLN-Zero |

## Results

Three model configurations were evaluated using noise-prediction MSE.

| Model               | Parameters |          MSE |
| ------------------- | ---------: | -----------: |
| Depth 6, AdaLN-Zero |     4.155M | **0.180605** |
| Depth 6, AdaLN      |     3.710M |     0.181090 |
| Depth 4, AdaLN-Zero |     2.822M |     0.182142 |

**Lower MSE indicates more accurate noise prediction on the evaluation set.**

### Observations

- The six-block AdaLN-Zero model achieved the lowest observed MSE.
- Increasing depth from four to six blocks reduced MSE by approximately 0.85%, while increasing the parameter count by approximately 47.2%.
- At depth six, AdaLN-Zero achieved approximately 0.27% lower MSE than ordinary AdaLN, with approximately 12% more parameters.

These differences are modest and come from individual training runs. They should not be interpreted as statistically significant without repeated experiments.

**Metric limitation:** Noise-prediction MSE is not directly comparable to the FID values reported in the original DiT paper. The dataset, model scale, resolution, and evaluation protocol also differ.

## Repository Structure

```text
Scalable-Diffusion-Transformers/
├── data/
│    └── cifar_latents.pt       # CIFAR-10 latent dataset
├── dit0.ipynb                  # CIFAR-10 preprocessing and VAE latent extraction
├── dit1.ipynb                  # DiT implementation, training, and evaluation
├── README.md
├── .gitignore
├── LICENSE
└── results
    ├── depth_4_adalnzero.png
    ├── depth6_adaln.png
    ├── depth6_adalnzero.png
    └── Pipeline.png
```

## Requirements

The implementation uses Python and PyTorch. Depending on the notebook cells, you may also need:

## Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/vaebhav10/Scalable-Diffusion-Transformers.git
   cd Scalable-Diffusion-Transformers
   ```

2. Install the required dependencies.

3. Open `dit0.ipynb` in Jupyter or upload it to Google Colab.
Running the Project

4. Generate CIFAR-10 latents

Open dit0.ipynb and run the preprocessing and VAE-encoding pipeline. This generates data/cifar_latents.pt, containing the latent representations and corresponding class labels (already included in the repo).

5. Train and evaluate DiT    
  
Open dit1.ipynb after generating the latent dataset. The notebook loads cifar_latents.pt, constructs the DiT architecture, trains the model, and evaluates the AdaLN and AdaLN-Zero variants at different depths.  

The reported experiments use 100,000 training steps per model.

**Requirements**

Install the dependencies used by the notebooks. The project uses PyTorch, torchvision, Hugging Face Diffusers, and supporting libraries for plotting and progress tracking.

## Limitations

- The model operates at a small latent resolution and has substantially fewer parameters than the large DiT configurations in the original paper.
- CIFAR-10 generation at this resolution limits the amount of visual detail that can be represented.
- A pretrained Stable Diffusion VAE may not be optimal for CIFAR-10 images.
- Noise-prediction MSE alone does not measure perceptual image quality.
- The reported comparisons use individual training runs and do not establish statistical significance.
- Training time and generated-image quality depend on the hardware, sampling procedure, and implementation details.

## Future Work

- Evaluate multiple random seeds to assess the stability of the ablation results.
- Compare generated samples and VAE reconstructions.
- Explore additional model depths, hidden dimensions, and patch sizes.
- Evaluate image quality with suitable generative metrics where feasible.
- Investigate faster sampling procedures and alternative latent representations.

## Acknowledgements

This project is based on the ideas introduced in *Scalable Diffusion Models with Transformers* by Peebles and Xie. It uses a pretrained Stable Diffusion VAE and the CIFAR-10 dataset.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
