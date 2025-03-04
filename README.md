# CycleGAN Monet Style Transfer

## Overview
This project implements a CycleGAN model to translate real-world photos into Monet-style paintings. The goal is to train a deep learning model that can generate realistic Monet-style artworks from natural images.

## Dataset
The dataset consists of:
- **Monet Paintings**: 300 images of Monet’s artwork.
- **Real Photos**: 7,038 landscape photographs.
- Images are provided in **256x256** resolution in both **JPEG** and **TFRecord** formats.

## Model Architecture
This project uses a **CycleGAN**, an unsupervised image-to-image translation model. The architecture consists of:
- **Two Generators**:
  - `G`: Translates photos → Monet paintings.
  - `F`: Translates Monet paintings → photos.
- **Two Discriminators**:
  - `D_M`: Classifies real vs. generated Monet paintings.
  - `D_P`: Classifies real vs. generated photos.

## Training Details
- **Loss Functions**:
  - **Adversarial Loss**: Helps generators create realistic images.
  - **Cycle Consistency Loss**: Ensures generated images can be transformed back.
  - **Identity Loss**: Encourages color preservation.
- **Optimizers**:
  -RMSprop optimizer with `learning_rate=1e-5`.
- **Batch Size**:
  - Default: `batch_size=1`
  - Optimized for GPU: `batch_size=2` or `4` (P100 GPU).

## Image Generation & Submission
- The trained CycleGAN generates **7,000+ Monet-style images**.
- The images are stored in a directory and **zipped** for submission.
- Final output: **`images.zip`** (single submission file).

## Steps to Train & Submit
1. **Train the CycleGAN model** 
2. **Generate Monet-style images** using the trained generator.
3. **Save images as JPG files** in `256x256` resolution.
4. **Zip the images** using `shutil.make_archive("images", 'zip', output_dir)`.
5. **Submit `images.zip`** on Kaggle.

## Challenges & Improvements
- **Training Stability**: 
  - Reduced discriminator updates to prevent overpowering the generator.
  - Tuned learning rates to improve convergence.
- **Mode Collapse**:
  - Adjusted loss function weights for better diversity in generated images.
- **Computation Time**:
  - Used **batch-based generation** for faster processing.
  - Optimized dataset pipeline with `tf.data`.

## Results
- Generated Monet-style paintings exhibit **color shifts and impressionist textures**.
- Further improvements needed for **fine details and brushstroke accuracy**.

## Future Work
- Train for **100+ epochs** for better quality.
- Use **higher resolution images** for more realistic output.
- Experiment with **different architectures** like StyleGAN or Diffusion Models.

## Acknowledgments
- Based on the **Kaggle "GANs Getting Started" competition**.
- Uses TensorFlow/Keras and `tf.data` for efficient training.
