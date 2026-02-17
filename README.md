# Breast Ultrasound Classification Project
## Tumor vs Normal vs Early Stage Tumor Detection

## Project Overview

This project focuses on multi-class classification of breast ultrasound images into:

- Normal Tissue
- Early-Stage Tumor
- Tumor Tissue

The study was conducted within the scope of an Image Processing course and aims to improve classification performance through systematic feature engineering and feature fusion techniques.

Total Dataset Size: **756 ultrasound images**

---

## Motivation

Initial experiments using only GLCM texture features achieved:

Accuracy: **68.78%**

This indicated that texture-only analysis was insufficient.

Therefore, a multi-feature architecture combining texture, shape, edge, statistical, and domain transformation features was developed.

---

## Preprocessing

- Grayscale conversion
- Speckle noise reduction
- Resize normalization
- Min-Max feature normalization (0–1 scaling)

---

## Feature Extraction Methods

### Texture Features
- GLCM (Contrast, Energy, Homogeneity, Correlation, ASM, Dissimilarity)
- LBP (Local Binary Pattern)
- LCP (Local Contrast Pattern)

### Shape & Geometry
- Area
- Perimeter
- Circularity
- Aspect Ratio

### Edge & Gradient
- HOG
- PHOG

### Statistical Features
- Mean
- Standard deviation
- Skewness
- Kurtosis

### Frequency Domain (Bonus)
- Fourier Transform (FFT)
- Wavelet Transform (LL, LH, HL, HH bands)

---

## Wavelet Domain Analysis (Bonus Study)

Wavelet transform decomposed images into:

- LL (Approximation)
- LH (Horizontal details)
- HL (Vertical details)
- HH (Diagonal details)

Best Single Band Performance:

LL Band + Random Forest → **70.63%**

All Wavelet Bands Combined → **75.39%**

This demonstrates that domain transformation improves robustness by isolating noise and preserving structural information.

---

## WEKA Classification Experiments

Algorithms used:

- Random Forest (100 & 500 trees)
- IBk (k=1)
- J48 Decision Tree

Evaluation Metrics:

- Accuracy
- Cohen’s Kappa

---

## Best Results

Feature Fusion (All Features Combined):

Random Forest Accuracy → **82.14%**

Cohen’s Kappa → **0.73**

This indicates substantial agreement beyond random classification.

---

## Key Observations

- Single features are insufficient for reliable diagnosis.
- Feature fusion significantly improves classification.
- Random Forest was the most stable and robust classifier.
- Wavelet transformation enhanced feature discriminability.
- Reduced feature selection achieved 79.10% accuracy with lower computational cost.

---

## Technologies

- Python
- OpenCV
- NumPy
- Scikit-image
- PyWavelets
- WEKA

---

## Course Information

Course: Image Processing  
Instructor: Aysun Sezer  
Developed as a final university project.

---

## Conclusion

This study demonstrates that hybrid feature engineering combined with ensemble classifiers can significantly improve breast ultrasound classification performance.

The proposed system achieved 82.14% accuracy and substantial reliability (Kappa = 0.73), making it a strong candidate for computer-aided diagnosis systems.
