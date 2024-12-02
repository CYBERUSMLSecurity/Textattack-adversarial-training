# Textattack-adversarial-training

This project aims to reproduce the TextAttack framework for adversarial attacks, data augmentation, and adversarial training in NLP, as described in the paper ["TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP"](https://arxiv.org/abs/2005.05909). 
Our objective is to validate the practical implementation of this framework by training models and generating adversarial examples, specifically the *Model Training* feature. The two notebooks in this project demonstrate different aspects of this implementation.

---

## **Files and Their Descriptions**

### 1. **Textattack_Train_fin.ipynb**
   - **Purpose:** This notebook focuses on training NLP models using the `textattack train` command, showcasing adversarial training and augmentation.
   - **Key Features:**
     - **Model Training:**
       - Trains an LSTM model on the Yelp Polarity dataset with 50 epochs and a batch size of 100.
       - Fine-tunes BERT on the GLUE CoLA task with 5 epochs and a batch size of 8.
     - **Adversarial Training:**
       - Demonstrates how adversarial attacks can be integrated during training to improve model robustness.
       - Includes examples using adversarial attacks like TextFooler and augmentation methods like Easy Data Augmentation (EDA).
     - **Why It's Important:**
       - This notebook validates the training functionality of TextAttack and ensures that it can generate adversarially robust models, aligning with the paper's objectives.

### 2. **Textattack-Train-fin-2.ipynb**
   - **Purpose:** This notebook extends the training experiments with additional models and datasets, emphasizing adversarial robustness.
   - **Key Features:**
     - **Model Training:**
       - Fine-tunes DistilBERT on the AG News dataset with adversarial training (TextFooler attack) and 2 clean epochs.
       - Trains RoBERTa on the Rotten Tomatoes dataset with EDA for data augmentation, including 4 transformations per example.
     - **Adversarial Examples:**
       - Showcases how adversarial attacks like TextFooler are used to create perturbations that flip model predictions.
     - **Why It's Important:**
       - The experiments in this notebook further validate the TextAttack framework's ability to generate adversarial examples and train models with enhanced robustness. This reinforces the practical utility of the methods described in the paper.

