# Plant Disease Classification with ResNet18 (PyTorch)

## Project Overview
This project builds an image classification model to detect plant diseases from leaf images using the [PlantVillage dataset](https://www.kaggle.com/) and **transfer learning** with a pre-trained ResNet18 model in PyTorch.

The goal is to help farmers identify plant diseases early, enabling targeted treatments and improved crop yields.

---

## Dataset
- **Source:** PlantVillage dataset (publicly available on Kaggle)
- **Images:** Thousands of labeled images of healthy and diseased leaves
- **Classes:** Multiple crops and diseases (e.g., Apple Scab, Citrus Greening, Powdery Mildew)
- **Structure:** Split into training and validation sets, with class-specific folders

---

## Data Preprocessing
- **Resize:** All images resized to 224×224 pixels to match ResNet18 input requirements
- **Normalization:** Using ImageNet mean and standard deviation values
- **Augmentation:** Random horizontal flips applied to training images for better generalization
- **Batching & Shuffling:**
  - Batch size: 32
  - Training data shuffled each epoch
  - Validation data not shuffled

---

## Model Architecture
- **Base Model:** ResNet18 pre-trained on ImageNet
- **Transfer Learning:** Final fully connected layer replaced with a layer matching the number of PlantVillage classes
- **Loss Function:** Cross Entropy Loss
- **Optimizer:** Adam with weight decay (`1e-4`) to reduce overfitting
- **Training Strategy:**
  1. **Warm-up phase:** Train only the final classification layer
  2. **Fine-tuning phase:** Unfreeze all layers and train at a lower learning rate

---

## Training Details
- **Device:** GPU (T4 on Google Colab)
- **Epochs:** Two-phase training (3 warm-up epochs + fine-tuning until early stopping)
- **Early Stopping:** Stops training if no improvement in validation accuracy for 3 consecutive epochs
- **Learning Rate Scheduling:** ReduceLROnPlateau based on validation loss

---

## Results

### Overall Metrics
- **Accuracy:** 92% on validation set (10,861 images)
- **Macro F1-score:** ~0.90  
- **Weighted F1-score:** ~0.92  

### Strengths
High performance on several major classes:
- Orange___Haunglongbing (F1 = 0.99)
- Soybean___healthy (F1 = 0.97)
- Squash___Powdery_mildew (F1 = 0.98)
- Corn___healthy (F1 = 0.99)
- Tomato___Tomato_Yellow_Leaf_Curl_Virus (F1 = 0.97)

### Weaknesses
Lower recall or precision on certain classes:
- Potato___healthy (Precision = 1.00, Recall = 0.52)
- Tomato___Early_blight (Precision = 0.76, Recall = 0.64)
- Corn___Cercospora_leaf_spot Gray_leaf_spot (F1 = 0.75)

### Class Imbalance
- Large classes (e.g., Orange___Haunglongbing, 1,102 samples) dominate accuracy.
- Small classes (e.g., Potato___healthy, 31 samples) contribute less to accuracy but still impact macro metrics.

---

## Confusion Patterns
- Misclassifications mainly occur between diseases with visually similar symptoms, such as different tomato blights.
- Healthy and diseased variants of the same crop sometimes overlap in predictions.

---

## Recommendations
1. **Augment minority classes** (Potato___healthy, Tomato___Early_blight) to improve recall.
2. **Targeted data augmentation** for difficult classes (simulate lighting changes, leaf damage variations).
3. **Adjust classification thresholds** to balance precision and recall for critical classes.


