# Animal Species Classification with EfficientNetB7

This project develops an image classification model to identify **7 different animal species** using **transfer learning** based on EfficientNetB7. The model is trained using TensorFlow/Keras and exported into multiple formats for deployment: **SavedModel**, **TFLite**, and **TensorFlow.js**.

---

## Model Overview

- **Classes**:  
  - 🦋 Butterfly  
  - 🐶 Dog  
  - 🐘 Elephant  
  - 🐔 Hen  
  - 🐒 Monkey  
  - 🕷 Spider  
  - 🐿 Squirrel

- **Architecture**:  
  - Base Model: `EfficientNetB7` (pre-trained on ImageNet)  
  - Head: GlobalAveragePooling2D + Dense layers  
  - Input Shape: 512 × 512 × 3

- **Training Strategy**:
  1. **Stage 1**: Train classification head (base model frozen)
  2. **Stage 2**: Fine-tuning (unfreeze top layers of base model)

- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Callbacks**:  
  - `EarlyStopping`  
  - `ReduceLROnPlateau`  
  - `ModelCheckpoint` (optional)  

---

## Model Directory Structure
├───tfjs_model
| ├───group1-shard1of1.bin
| └───model.json
├───tflite
| ├───model.tflite
| └───label.txt
├───saved_model
| ├───saved_model.pb
| └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt

---

## Requirements
pip install -r requirements.txt

---

## Author

Mohammad Al Muktabar
