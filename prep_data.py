import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split

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
    lang_data = lang_data[['text', 'binary_labels', 'original_labels']]
    lang_train, lang_test = train_test_split(lang_data, test_size=0.2)
    lang_test, lang_dev = train_test_split(lang_test, test_size=0.5)
    print(len(lang_train), len(lang_dev), len(lang_test))
    pk.dump(lang_train.to_dict(orient='list'), open(data_dir + f"clean_{lang_code_map[lang]}_train.pk", "wb"))
    pk.dump(lang_test.to_dict(orient='list'), open(data_dir + f"clean_{lang_code_map[lang]}_test.pk", "wb"))
    pk.dump(lang_dev.to_dict(orient='list'), open(data_dir + f"clean_{lang_code_map[lang]}_dev.pk", "wb"))