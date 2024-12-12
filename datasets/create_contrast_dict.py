import jsonlines
import nltk
nltk.download('wordnet')
import spacy

nlp = spacy.load('en_core_web_sm')

# this code literally just prints the nltk synsets for various words
# then, we parsed through them manually and created a dictionary of acceptable replacements
def get_word_replacements(sentence):
    doc = nlp(sentence)
    options = {}
    
    pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
    
    for token in doc:
        if token.pos_ in pos:
            synsets = nltk.corpus.wordnet.synsets(token.text)
            alts = set()
            
            for s in synsets:
                # only allow top 5 most common words to be considered
                if s == synsets[0] or s == synsets[1] or s == synsets[2] or s == synsets[3] or s == synsets[4]:  
                    for lemma in s.lemmas():
                        if lemma.name().lower() != token.text.lower():
                            alts.add(lemma.name())
            
            if alts:
                options[token.text] = list(alts)
    
    return options

# test_data = []
# with jsonlines.open('./datasets/snli_1.0/snli_1.0_dev.jsonl') as reader:
#     for i, obj in enumerate(reader):
#         if i < 50:
#             test_data.append(obj)
#         else:
#             break

# print("Analyzing potential replacements for 5 examples:")
# for idx, example in enumerate(test_data):
#     print(f"\nExample {idx + 1}:")
    
#     print("\nOriginal premise:", example['sentence1'])
#     print("Original hypothesis:", example['sentence2'])
#     print("Label:", example['gold_label'])
#     hypothesis_options = get_word_replacements(example['sentence2'])

#     print("\nPossible replacements for hypothesis:")
#     for word, replacements in hypothesis_options.items():
#         print(f"'{word}' could be replaced with: {replacements}")
#     print("-" * 50)

manual_replacements = {
    # colors
    'blue': ['azure', 'cerulean',],
    'red': ['crimson', 'scarlet',],
    'green': ['emerald','lime-green',],
    'yellow': ['amber',],
    'orange': ['tangerine',],
    'purple': ['violet', 'lavender',],
    'pink': ['magenta',],
    'brown': ['sepia', 'coffee-colored',],

    # verbs
    'washing': ['cleaning', 'scrubbing',],
    'walking': ['strolling', 'striding',],
    'hugging': ['embracing','cuddling',],
    'goodbye': ['bye', 'farewell',],
    'eating': ['consuming',],
    'jumping': ['leaping',],
    'dancing': ['prancing',],
    'waiting': ['anticipating',],
    'leaving': ['departing',],
    'catching': ['capturing',],

    # people
    'women': ['ladies','females'],
    'woman': ['lady','female',],
    'sister': ['sibling',],
    'men': ['gentlemen','males'],
    'man': ['gentleman','male',],
    'kids': ['children', 'youngsters',],
    'children': ['kids', 'youngsters',],
    'child': ['kid', 'youngster',],
    'boy': ['child','male child',],
    'girl': ['child','female child',],
    # 'hands': ['palms', 'fingers',],
    'person': ['individual',],
    'senior': ['elderly individual',],

    # professions
    'doctor': ['physician','medic',],
    'doctors': ['physicians','medics',],
    'teacher': ['educator',],
    'teachers': ['educators',],
    'scientist': ['researcher',],
    'scientists': ['researchers',],
    'artist': ['painter',],
    'musician': ['instrumentalist',],
    'singer': ['vocalist',],

    # random nouns
    'car': ['automobile',],
    'road': ['street',],
    'surgery': ['operation',],
    'camera': ['photographing device',],
    'food': ['nourishment','meal',],
    'movie': ['film',],
    'bicyle': ['bike','cycle','two-wheeler',],
    'book': ['novel',],
    'phone': ['telephone',],
    'computer': ['laptop',],
    'house': ['home',],
    'dog': ['canine',],
    'cat': ['feline',],
    'plane': ['airplane',],
    'shirt': ['garment',],
    'shoes': ['footwear',],


    # adjs
    'long': ['lengthy',],
    'short': ['brief',],
    'big': ['large',],
    'small': ['little',],
    'happy': ['joyful',],
    'sad': ['unhappy',],
    'angry': ['furious',],
    'tall': ['high',],
    'short': ['low',],
    'old': ['elderly',],
    'young': ['youthful',],
    'very': ['extremely',],
    'bad': ['subpar',],
    'good': ['excellent',],
}

import json
with open('manual_replacements.json', 'w') as f:
    json.dump(manual_replacements, f, indent=4)
