import jsonlines

def contrast_test(input_path, output_path):
    contrast_dataset = []

    label_map = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    
    with jsonlines.open(input_path) as reader:
        for obj in reader:
            if obj['gold_label'] == '-':
                continue
            
            contrast_example = {
                'premise': obj['sentence1'],
                'hypothesis': obj['sentence2'],
                'label': label_map[obj['gold_label']]
            }
            contrast_dataset.append(contrast_example)
    
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(contrast_dataset)
    
    return len(contrast_dataset)

input_file = 'snli_1.0/snli_1.0_test.jsonl'
output_file = 'snli_1.0/snli_contrast_test.jsonl'
total = contrast_test(input_file, output_file)
print(f"Created contrast test set with {total} examples")
