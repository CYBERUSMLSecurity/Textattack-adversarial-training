# # TextAttack Model Fine-Tuning with Data Augmentation

This project aims to reproduce the TextAttack framework for adversarial attacks, data augmentation, and adversarial training in NLP, as described in the paper ["TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP"](https://arxiv.org/abs/2005.05909). 
Our objective is to validate the practical implementation of this framework by training models and generating adversarial examples, specifically the *Adversarial Model Training* feature. The script.py in this project demonstrates implementation of the same. 
- This project demonstrates fine-tuning a **RoBERTa-based model** for sentiment classification using the **Rotten Tomatoes** dataset. The fine-tuning process leverages **Easy Data Augmentation (EDA)** techniques from the **TextAttack** library to improve model generalization.
---

## **Files and Their Descriptions**

### 1. **script.py**  
This Python script implements the evaluation pipeline for adversarial attacks on a machine learning model trained on the CIFAR-10 dataset. It includes loading the model, generating adversarial examples, and assessing the model's performance.

### 2. **results.txt**  
A plain text file that records the outcomes of adversarial attacks, including success rates and performance metrics, for different attack methods used in the evaluation*


# TextAttack Model Fine-Tuning with Data Augmentation

This project demonstrates fine-tuning a **RoBERTa-based model** for sentiment classification using the **Rotten Tomatoes** dataset. The fine-tuning process leverages **Easy Data Augmentation (EDA)** techniques from the **TextAttack** library to improve model generalization. 

## Overview

The primary goals of this project are:

- Fine-tune the `roberta-base` model for a binary sentiment classification task (positive or negative reviews).  
- Enhance the robustness of the model by augmenting the training data using EDA.  
- Evaluate model performance through training and testing accuracy, loss, and generalization trends.

---

## Environment Setup

To run the training script, ensure you have the following installed:

- Python 3.10+
- TextAttack library  
- PyTorch and HuggingFace Transformers  
- NLTK for text augmentation  

Install the required packages using the following commands:

```bash
pip install textattack
pip install transformers
pip install torch torchvision
pip install nltk
```

Ensure a CUDA-compatible GPU is configured for optimal performance.

---

## Training Pipeline

### Dataset
The **Rotten Tomatoes** dataset is used for training and evaluation. It contains two splits:
- **Train Split**: 8,530 examples  
- **Test Split**: 1,062 examples  

### Model
The model used is `roberta-base` from the HuggingFace Transformers library, fine-tuned with a classification head for binary classification.

### Data Augmentation
**Easy Data Augmentation (EDA)** is applied with the following settings:
- **Percentage of words swapped per text**: 10%  
- **Transformations per example**: 4  

### Training Configuration
- **Number of epochs**: 5  
- **Learning rate**: 1e-5  
- **Batch size**: 8 (per device)  
- **Optimizer**: AdamW (from HuggingFace)  

---

## Training Results

### Key Metrics
- **Train Accuracy**: Improved from ~78% to ~99.94% over five epochs.  
- **Evaluation Accuracy**: Stabilized at ~88-89% with slight fluctuations over epochs.

### Observations
1. **Loss Reduction**: Consistent reduction in loss, indicating effective learning.  
2. **Train vs Eval Accuracy Gap**: Highlights a potential overfitting trend due to the small dataset size. EDA mitigates this but does not eliminate it entirely.  
3. **Best Epoch**: Epoch 4 yielded the highest evaluation accuracy at 89.12%.  

---

## **Customization Details**

In this project, the following customizations were chosen for specific reasons:

### **EDA Parameters**
- **`pct_words_to_swap`**: Set to **0.1** to ensure 10% of the words in each input are swapped. This strikes a balance between maintaining semantic coherence and introducing diversity for model robustness.
- **`transformations_per_example`**: Set to **2** to generate two augmented examples per input, effectively expanding the dataset while avoiding overfitting.

### **Training Arguments**
- **Epochs**: Configured to **5**, providing sufficient time for the model to learn patterns without overtraining or excessive computational overhead.
- **Learning Rate**: Set to **0.001**, a standard value for stable convergence in deep learning tasks.
- **Batch Size**: Chosen as **32** to optimize GPU usage and ensure manageable memory consumption during training.

These parameters were selected to balance computational efficiency and robust evaluation of adversarial attacks.

---

## Outputs

The following outputs are generated:

1. **Logs**: Stored in `outputs/train_log.txt`.  
---

## How to Run

To reproduce the training pipeline, run the following command:

- Ensure you have python3.10 version installed
```bash
python3 --version
```
- Create a python3.10 virtual environment to run the attacks
```bash
python3.10 -m venv textattack-venv310
```

- Activate the env
```bash
source textattack-venv/bin/activate
```

_ Install all the above packages
```bash
pip install textattack transformers torch pandas
```

- Run the script 

```bash
python3 script.py
```

- After running, it will output an `outputs/`
folder. 

### Customization
You can customize the script by modifying:
- **EDA Parameters**: `pct_words_to_swap` and `transformations_per_example`.  
- **Training Arguments**: Epochs, learning rate, or batch size.

---

## Challenges Faced

1] numpy python dependecy [3.12] - Apparently the numpy package is not compatuble with the 3.12 version of python which gave errors while running the script
   Solution - After a while struggling to find solution I found a solution at https://github.com/google/sentencepiece/issues/971 which suggested to downgrade the python version to 3.10 which eventually helped.

---

## Future Work
1. Experiment with adversarial training techniques available in TextAttack.  
2. Fine-tune other transformer-based models (e.g., BERT, DistilBERT).  
3. Explore additional data augmentation methods to mitigate overfitting further.  

---

## Citation
If you use this project, consider citing the libraries:  

- **HuggingFace Transformers**:  
  > [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

- **TextAttack**:  
  > Morris, John X., et al. "TextAttack: A Framework for Adversarial Attacks in Natural Language Processing." 
