from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

prediction_args = TrainingArguments(
    output_dir="./predictions",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=32,
    dataloader_drop_last=False
)

trainer = Trainer(
    model=model,
    args=prediction_args,
)

contrast_dataset = 'datasets/snli_1.0/snli_contrast_final.jsonl'
predictions = trainer.predict(contrast_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)

for i, example in enumerate(contrast_dataset):
    print(f"\nPremise: {example['premise']}")
    print(f"Hypothesis: {example['hypothesis']}")
    
    # 0: entailment, 1: neutral, 2: contradiction
    print(f"Predicted Label: {pred_labels[i]}") 
