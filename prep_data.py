import pandas as pd
import pickle as pk

data_dir = './data/'
data = pd.read_csv(data_dir + "train.csv")

lang_code_map = {
    "English": "en", 
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "French": "fr",
    "Chinese": "zh"
}

for lang in data['language'].unique():
    lang_data = data[data['language'] == lang].copy()
    lang_data['binary_labels'] = lang_data['label'].apply(lambda l: 0 if l<=3 else 1)
    lang_data['original_labels'] = lang_data['label']
    lang_data = lang_data[['text', 'binary_labels', 'original_labels']].to_dict()
    output_filename = f"clean_{lang_code_map[lang]}.pk"
    pk.dump(lang_data, open(data_dir + output_filename, "wb"))
    