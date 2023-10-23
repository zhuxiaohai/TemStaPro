import os

import torch
from transformers import T5EncoderModel, EsmModel

from MLP import MLP_C2H2


class TemStaProModel(torch.nn.Module):
    def __init__(self, parameters, options):
        super(TemStaProModel, self).__init__()

        self.config = parameters
        self.per_res_mode = (options.per_res_out or options.per_segment_out)
        self.pt_model = self.get_pretrained_model(options.pt_dir)

        self.classifier = MLP_C2H2(parameters["INPUT_SIZE"],
                                   parameters["HIDDEN_LAYER_SIZES"][0],
                                   parameters["HIDDEN_LAYER_SIZES"][1],
                                   activation=parameters["ACTIVATION"])

        if parameters["LOAD_PRETRAINED_CLASSIFIER"]:
            model_path = "%s/%s_%s_%s-%s_s%s.pt" % (
                parameters["CLASSIFIERS_DIR"], parameters["EMB_TYPE"],
                parameters["DATASET"], parameters["CLASSIFIER_TYPE"],
                parameters["THRESHOLD"], parameters['SEED'])
            self.classifier.load_state_dict(torch.load(model_path))
            print('loading pretrained classifier: done')

    def get_pretrained_model(self, model_path):
        pass

    def get_embeddings(self, batch):
        results = {'per_res_representations': dict(),
                   'mean_representations': dict()}
        # pdb_ids = batch['pdb_id']
        input_idx = batch['input_ids']
        attention_mask = batch['attention_mask']
        # pdb_ids = batch['pdb_id']
        # input_idx = batch
        # attention_mask = torch.ones(batch.shape).cuda(batch.device)
        # seq_lens = batch['seq_len']

        with torch.no_grad():
            embedding_repr = self.pt_model(input_idx, attention_mask=attention_mask)

        results = (embedding_repr.last_hidden_state * attention_mask.unsqueeze(-1)).sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(-1)

        # for batch_idx, identifier in enumerate(pdb_ids):
        #     emb = embedding_repr.last_hidden_state[batch_idx, :attention_mask[batch_idx].sum()-1]
        #     if self.per_res_mode:
        #         results["per_res_representations"][identifier] = emb.squeeze()
        #     protein_emb = emb.mean(dim=0)
        #     results["mean_representations"][identifier] = protein_emb.squeeze()

        return results

    def forward(self, data):
        # Generating embeddings
        embeddings = self.get_embeddings(data)

        # input_data = None
        # for i, seq_id in enumerate(embeddings["mean_representations"].keys()):
        #     embedding = embeddings["mean_representations"][seq_id]
        #     if (i):
        #         input_data = torch.vstack((input_data, torch.flatten(embedding)))
        #     else:
        #         input_data = torch.reshape(embedding, (1, self.config["INPUT_SIZE"]))
        # Generating outputs
        if self.config["ONLY_EMBEDDINGS"]:
            return embeddings
        else:
            outputs = self.classifier(embeddings)
            return outputs

    def train(self, mode: bool = True):
        super(TemStaProModel, self).train(mode)
        self.pt_model.eval()


class T5BaseModel(TemStaProModel):
    def get_pretrained_model(self, model_path):
        if (os.path.exists(model_path + '/pytorch_model.bin') and
                os.path.exists(model_path + '/config.json')):
            model = T5EncoderModel.from_pretrained(model_path + '/pytorch_model.bin',
                                                   config=model_path + '/config.json')
        else:
            model = T5EncoderModel.from_pretrained(model_path)
        model = model.eval()

        return model


class ESMBaseModel(TemStaProModel):
    def get_pretrained_model(self, model_path):
        model = EsmModel.from_pretrained(model_path, add_pooling_layer=False)
        model = model.eval()
        return model