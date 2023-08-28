import os

import pandas as pd
from torch.utils.data import Dataset
from prottrans_models import get_tokenizer


class TemStaProData(Dataset):
    def __init__(self, csv_path, pt_dir, pt_server_path, pdb_id='idx', label='stability_at_40'):
        super(TemStaProData).__init__()
        self.label = label
        self.pdb_id = pdb_id
        self.df = pd.read_csv(csv_path, encoding='utf8', sep=',')
        self.max_length = self.df['sequence'].str.len().max() + 1
        self.df[self.label] = self.df[self.label].astype(int)
        self.pt_dir = pt_dir
        self.pt_server_path = pt_server_path
        self.load_tokenizer()
        self.pdb_ids = self.df[self.pdb_id].tolist()
        self.label = self.df[self.label].tolist()
        self.input_data = self.df['sequence'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # pdb_id, input_data, label = self.df[[self.pdb_id, 'sequence', self.label]].iloc[idx]
        # token_encoding = self.tokenizer.encode_plus(' '.join(list(input_data)),
        #                                             add_special_tokens=True,
        #                                             padding='max_length',
        #                                             max_length=self.max_length,
        #                                             return_tensors='pt')
        # return label, {'pdb_id': pdb_id, 'input_idx': token_encoding['input_ids'].squeeze(0),
        #                'attention_mask': token_encoding['attention_mask'].squeeze(0)}
        return self.label[idx], ' '.join(list(self.input_data[idx]))

    def load_tokenizer(self):
        """
        Load ProtTrans model and tokenizer.

        pt_dir - STRING to determine the path to the directory with ProtTrans
            "pytorch_model.bin" file
        pt_server_path - STRING of the path to ProtTrans model in its server

        returns (ProtT5-XL model, tokenizer)
        """
        if (not os.path.exists(f"{self.pt_dir}/")):
            os.system(f"mkdir -p {self.pt_dir}/")

        if (os.path.isfile(f"{self.pt_dir}/tokenizer_config.json")):
            # Only loading the tokenizer
            tokenizer = get_tokenizer(self.pt_dir)
        else:
            # Downloading and saving the tokenizer
            tokenizer = get_tokenizer(self.pt_server_path)
            tokenizer.save_pretrained(self.pt_dir)

        self.tokenizer = tokenizer


