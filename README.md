# Application-of-GANs-to-microscopy-image-for-cell-culture-counting
Pix2Pix GAN in PyTorch for automated cancer cell colony segmentation from microscopy images. Includes custom dataset pipeline, GPU training, evaluation metrics (IoU, Dice, F1), and visualization of real vs generated outputs, reducing manual annotation effort by 80%.  It supports a full pipeline from manual annotations to image augmentation, dataset generation, model training, and evaluation for image-to-image translation tasks.

---

## Required Libraries

```bash
pip install numpy opencv-python matplotlib torch torchvision scikit-learn scipy pandas
```
if you did miss any libraries please install using pip
please install fiji ImageJ for 
## How to Run

1. Open `Automation_Script.ipynb` in Jupyter or Colab.
2. Run the first cell to convert into Black_filled_dataset
3. Use the Black_filled_dataset as input to fiji `Create_GroundTruth_Image.ijm` macro and get the ground truth dataset (Dataset Output)
4. Then use the image agumentation cell to create synthetic dataset (orelse you may face overfitting while training pix2pix)
5. Combine all the agumented images into one folder and split each images into 16 to avoid lose of image quality
6. Then conver the all the .tiff files to .jpg for pix2pix model training
7. Then randomly split the data into train and val and run the pix2pix model traing
8. Run Evaluation metrices 
9. Then run `particle_Aanalyse.ijm` in fiji to fetch all the necessary statistics.


---

## Full Pipeline Overview

1. **Annotation Preprocessing**  
   Manually annotated `.tiff` images with cyan regions are converted into clean binary masks using a Python script, saved in `Black_Filled_dataset/`.

2. **Ground Truth Generation (Fiji Macro)**  
   The `Black_Filled_dataset/` is processed in Fiji using the macro `Create_GroundTruth_Image.ijm`. The output binary masks are saved in `Dataset_Output/`.

3. **Input-Ground Truth Pairing**  
   Input images from `Dataset/` and their corresponding masks from `Dataset_Output/` are combined side-by-side and saved in `Combined_Folder/` for Pix2Pix training.

4. **Data Augmentation**  
   Augmentation techniques such as rotation, brightness variation, and gradient brightness are applied. The results are saved in:
   - `Modified_random_rotation/`
   - `Modified_brightness/`
   - `Modified_gradient_brightness/`

5. **Combining Augmented and Original Images**  
   All augmented folders and the original `Combined_Folder/` are merged into `Combined_Augumented_Folder/`.

6. **Tiling to 256x256**  
   Each image in `Combined_Augumented_Folder/` is split into 16 tiles of 256x256 pixels, saved in `patches/`.

7. **TIFF to JPG Conversion**  
   The tiled images are converted from `.tiff` to `.jpg` format for compatibility with the training process.

8. **Train/Validation Split**  
   The dataset is split into training and validation sets for training the Pix2Pix model.

9. **Pix2Pix Model Training**  
   The model is trained for 100 epochs. Loss graphs for generator and discriminator are plotted, and the best checkpoint is saved in `best_model/`.

10. **Evaluation Metrics**  
   The trained model is evaluated using several metrics (see below). Visual results and error cases are saved in `evaluation_outputs/`.

11. **Particle Analysis (Fiji Script)**  
   The predicted segmentation outputs are passed through the Fiji macro `particle_Aanalyse.ijm` to extract final statistics such as total colony count, area, and distribution.

---


## Folder Structure Overview

```
.
├── Black_Filled_dataset/           # Masks with cyan areas filled black
├── Combined_Folder/                # Side-by-side input + mask images (pix2pix format)
├── Combined_Augumented_Folder/     # Final training data with augmentations
├── Dataset/                        # Training images
├── Dataset_Output/                 # Ground truth masks
├── Manually_annotated_dataset/     # Raw annotated microscopy images (.tiff)
├── Modified_brightness/            # Augmented with brightness changes
├── Modified_gradient_brightness/   # Augmented with gradient illumination
├── Modified_random_rotation/       # Rotated image augmentations
├── patches/                        # 256x256 patches split from large images
├── Final_Script.ipynb              # Main notebook (preprocessing, training, evaluation)
├── Create_GroundTruth_Folder.ijm   # ImageJ macro script for ground truth folder creation
├── Create_GroundTruth_Image.ijm    # ImageJ macro for mask creation
├── particle_analyse.ijm      # ImageJ macro for post-processing and counting
```

## Pipeline Summary

| Stage            | Description                                                                       |
|------------------|-----------------------------------------------------------------------------------|
| Preprocessing     | Cyan-filled masks are converted into clean binary masks.                         |
| Image Pairing     | Input and mask images are combined side-by-side for Pix2Pix training.            |
| Augmentation      | Brightness, gradient, and rotation augmentations are applied.                    |
| Tiling            | Images are split into 256x256 patches stored in `patches/`.                      |
| Training          | Pix2Pix GAN using U-Net generator and PatchGAN discriminator.                    |
| Evaluation        | Performance metrics and colony count analysis with error visualization.          |

---


## ImageJ Macros

This project includes the following ImageJ macros:

- `Create_GroundTruth_Image.ijm`: Converts annotated images into binary masks.
- `Particle_Analyse_.ijm`: Performs colony counting and analysis on the predicted outputs.

These should be run using Fiji (ImageJ).

---

## Tips

- `.tiff` files are expected to be grayscale or contain cyan annotations.
- Ensure consistent naming between input images and masks.
- You may merge all `Modified_*` folders with the original dataset before model training.
