import json
import random
import jsonlines

with open('manual_replacements.json', 'r') as f:
    replacements = json.load(f)

# replace words in a sentence with synonyms, up to a certain threshold
# (5 minutes' worth of) empirical testing says 0.35 is a good threshold
def replace_words(sentence, threshold=0.35):
    words = sentence.split()
    new_sentence = []
    max_replaced = int(len(words) * threshold)
    num_replaced = 0
    mods = []
    for word in words:
        if word.lower() in replacements and num_replaced < max_replaced:
            new_word = random.choice(replacements[word.lower()])
            # maybe consider not changing proper nouns at all?
            if word[0].isupper():
                new_word = new_word.capitalize()
            new_sentence.append(new_word)
            num_replaced += 1
            mods.append((word, new_word))
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence), mods

# test_sentences = [
#     "The blue car is waiting for the woman.",
#     "A young boy is walking with his dog.",
#     "The scientist is conducting an experiment.",
#     "She is wearing a red shirt and blue shoes."
# ]

# for sentence in test_sentences:
#     print("\nOriginal:", sentence)
#     print("Modified:", replace_words(sentence, loaded_replacements))

import nltk
from nltk.tree import Tree

def clean_formatting(text):
    return ' '.join(text.split())

def fix_tree(tree_str, mods, binary=False):
    word_map = dict(mods)
    
    def replace_word(word):
        if word.lower() in [old.lower() for old in word_map.keys()]:
            old_word = next(old for old in word_map.keys() if old.lower() == word.lower())
            new_word = word_map[old_word]
            if word[0].isupper():
                new_word = new_word.capitalize()
            return new_word
        return word
    
    if binary:
        # easier case, j treat as a string
        new_tokens = [replace_word(token) for token in tree_str.split()]
        return clean_formatting(' '.join(new_tokens))
    else:
        # use nltk library to handle POS tags
        tree = Tree.fromstring(tree_str)
        leaves = tree.treepositions('leaves')
        for idx in leaves:
            tree[idx] = replace_word(tree[idx])
        return clean_formatting(str(tree))

def create_snli_contrast(input_path, output_path):
    contrast_dataset = []
    # i = 0
    with jsonlines.open(input_path) as reader:
        for obj in reader:
            # i += 1
            # if i == 10:
            #     break

            if obj['gold_label'] == '-':
                continue
                
            # maintaining SNLI structure for ease of use
            contrast_example = obj.copy()
            
            new_hypothesis, mods = replace_words(obj['sentence2'])
            contrast_example['sentence2'] = new_hypothesis

            # we also have to update parse trees i think
            if 'sentence2_parse' in obj:
                contrast_example['sentence2_parse'] = fix_tree(
                    obj['sentence2_parse'], 
                    mods
                )
            
            if 'sentence2_binary_parse' in obj:
                contrast_example['sentence2_binary_parse'] = fix_tree(
                    obj['sentence2_binary_parse'], 
                    mods,
                    binary=True
                )
            
            contrast_dataset.append(contrast_example)

    
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(contrast_dataset)
    
    return len(contrast_dataset)

input_file = 'snli_1.0/snli_1.0_dev.jsonl'
output_file = 'snli_1.0/snli_contrast_dev.jsonl'
total = create_snli_contrast(input_file, output_file)
print(f"Created {total} contrast examples")


