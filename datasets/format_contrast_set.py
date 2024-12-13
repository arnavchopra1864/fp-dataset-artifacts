import jsonlines

def convert_snli_format(input_path, output_path):
    label_map = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    converted_data = []
    
    with jsonlines.open(input_path) as reader:
        for obj in reader:
            if obj['gold_label'] == '-':
                continue
                
            converted_example = {
                'premise': obj['sentence1'],
                'hypothesis': obj['sentence2'],
                'label': label_map[obj['gold_label']]
            }
            converted_data.append(converted_example)
    
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(converted_data)
    
    return len(converted_data)

input_file = './snli_1.0/snli_1.0_dev.jsonl'
output_file = './snli_1.0/snli_1.0_converted.jsonl'
total = convert_snli_format(input_file, output_file)
print(f"Converted {total} examples")
