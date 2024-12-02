import transformers
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack import Trainer
from textattack.augmentation import EasyDataAugmenter
from textattack.training_args import TrainingArgs

# Load the datasets
train_dataset = HuggingFaceDataset("rotten_tomatoes", split="train")
eval_dataset = HuggingFaceDataset("rotten_tomatoes", split="test")  # Use the test split for evaluation

# Load the RoBERTa model and tokenizer
model = transformers.AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")

# Wrap the model using HuggingFaceModelWrapper
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Initialize Easy Data Augmentation
eda_augmenter = EasyDataAugmenter(pct_words_to_swap=0.1, transformations_per_example=4)

# Define training arguments
training_args = TrainingArgs(
    num_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=8,  # Adjust batch size as needed
)

# Create a trainer instance with task type set to "classification"
trainer = Trainer(model_wrapper, "classification", train_dataset=train_dataset, eval_dataset=eval_dataset, training_args=training_args)

# Fine-tune the model with augmentations for each epoch
for epoch in range(training_args.num_epochs):
    print(f"Starting epoch {epoch + 1}/{training_args.num_epochs}")
    
    # Generate augmented data for training
    augmented_data = []
    
    # Iterate over the dataset examples (tuples)
    for example in train_dataset:
        # Accessing the text input directly from the tuple
        text = example[0]  # Assuming first element is the input text
        
        # Generate augmented examples using EDA
        augmented_examples = eda_augmenter.augment(text)
        augmented_data.extend(augmented_examples)
    
    # You may want to create a new dataset or combine it with original data here.
    
    # Train on the original dataset (or augmented dataset if you choose to update)
    trainer.train()

print("Training complete.")
