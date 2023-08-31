# coding: utf-8
import argparse
import numpy as np
import random
import pickle
import yaml
from easydict import EasyDict
from optparse import OptionParser
import sys
import os

import torch
from torch.utils.data import DataLoader

from dataset import TemStaProData
import runner
import utils
import models
import prottrans_models
from prottrans_models import get_tokenizer


PARAMETERS = {
    "PT_MODEL_PATH": "Rostlab/prot_t5_xl_half_uniref50-enc",
    "ACTIVATION": False,
    "LOAD_PRETRAINED_CLASSIFIER": False,
    "THRESHOLD": "40",
    "SEED": "41",
    "DATASET": "major",
    "EMB_TYPE": "mean",
    "CLASSIFIER_TYPE": "imbal",
    "THRESHOLDS": {
        ":(40-65]:": ["40", "45", "50", "55", "60", "65"],
        ":(40-80]:": ["40", "45", "50", "55", "60", "65", "70", "75", "80"],
    },
    "SEEDS": ["41", "42", "43", "44", "45"],
    "INPUT_SIZE": 1024,
    "HIDDEN_LAYER_SIZES": [512, 256],
    "THRESHOLDS_RANGE": ":(40-65]:",
    "TEMPERATURE_RANGES": {
        ":(40-65]:": ["<40", "[40-45)", "[45-50)", "[50-55)", "[55-60)",
            "[60-65)", "65<="],
        ":(40-80]:": ["<40", "[40-45)", "[45-50)", "[50-55)", "[55-60)",
            "[60-65)", "[65-70)", "[70-75)", "[75-80)", "80<="],
    },
    "THERMOPHILICITY_LABELS": {
        "mesophilic": ["<40", "[40-45)", "<45"],
        "thermophilic": ["[45-50)", "[50-55)", "[55-60)", "[60-65)",
            "65<=", "[65-70)", "[70-75)", "<75"],
        "hyperthermophilic": ["[75-80)", "80<="]
    },
    "PRINT_THERMOPHILICITY": {
        ":(40-65]:": False,
        ":(40-80]:": True
    }
}

parser = OptionParser()

parser.add_option('--config_path', type=str, help='path of config',
                    default='/home/zhuxh/PycharmProjects/TemStaPro/config/temstat.yml')

parser.add_option("--input-fasta", "-f", dest="fasta",
    default='./tests/data/long_sequence.fasta', help="path to the input FASTA file.")

parser.add_option("--embeddings-dir", "-e", dest="emb_dir",
    default='tests/outputs/', help="path to the directory to which embeddings "+\
    "files will be saved (cache).")

parser.add_option("--PT-directory", "-d", dest="pt_dir",
    default='./ProtTrans/', help="path to the directory of ProtTrans model.")

parser.add_option("--temstapro-directory", "-t", dest="tsp_dir",
    default='./', help="path to the directory of TemStaPro program "+\
    "with its dependencies.")

parser.add_option("--more-thresholds", dest="more_thresholds",
    action="store_true", help="option for the mode that outputs "+\
    "additional predictions for upper temperature thresholds and the "+\
    "thremophilicity label")

parser.add_option("--mean-output", dest="mean_out",
    default='./long_sequence_predictions.tsv', help="path to the output TSV file with mean predictions. "+\
    "Predictions made from the mean embeddings are always printed to STDOUT."+\
    " If this option is given, the output is directed to the given file")

parser.add_option("--per-res-output", dest="per_res_out",
    default=None, help="path to the output TSV file with per-residue "+\
    "predictions.")

parser.add_option("--per-segment-output", dest="per_segment_out",
    default=None, help="path to the output TSV file with per-residue "+\
    "predictions made for each segment of the sequence.")

parser.add_option("--segment-size", dest="segment_size",
    default=41, help="option to set the window size for average smoothening "+\
    "of per residue embeddings ('per-segment-output' option). Default: 41.")

parser.add_option("--window-size-predictions", "-w",
    dest="window_size_predictions",
    default=81, help="option to set the window size for average smoothening "+\
    "of per residue predictions for plotting (option for 'per-res-output' "+\
    "and 'per-segment-output'). Default: 81.")

parser.add_option("--per-residue-plot-dir", "-p", dest="plot_dir",
    default=None, help="path to the directory to which inferences "+\
    "plots will be saved (option for 'per-res-output' and "+\
    "'per-res-segment-output' modes. Default: './'.")

parser.add_option("--curve-smoothening", "-c", dest="curve_smoothening",
    default=False, action="store_true",
    help="option for 'per-segment-output' run mode, which adjusts the "+\
    "plot by making an additional smoothening of the curve.")

parser.add_option("--portion-size", dest="portion_size",
    default=1000,
    help="option to set the portions', into which to divide the input "+\
    "of sequences, maximum size. If no division is needed, set the "+\
    "option to 0. Default: 1000.")

parser.add_option("--version", "-v", dest="version",
    default=False, action="store_true",
    help="print version of the program and exit.")

(options, args) = parser.parse_args()

if(options.version):
    print(f"TemStaPro 0.1.{os.popen('git rev-list --all --count').read().strip()}")
    exit()

options.window_size_predictions = int(options.window_size_predictions)
options.segment_size = int(options.segment_size)
if(options.more_thresholds): PARAMETERS['THRESHOLDS_RANGE'] = ":(40-80]:"

try:
    assert (options.fasta != None), f"{sys.argv[0]}: a FASTA file is required."
except AssertionError as message:
    print(message, file=sys.stderr)
    exit()

try:
    assert (options.pt_dir != None), (
        f"{sys.argv[0]}: a path to the ProtTrans model location is required."
    )
except AssertionError as message:
    print(message, file=sys.stderr)
    exit()

PARAMETERS["CLASSIFIERS_DIR"] = f"{options.tsp_dir}/models"

per_res_mode = (options.per_res_out or options.per_segment_out)

if (not os.path.exists(f"{options.pt_dir}/")):
    os.system(f"mkdir -p {options.pt_dir}/")

if (os.path.isfile(f"{options.pt_dir}/tokenizer_config.json")):
    # Only loading the tokenizer
    tokenizer = get_tokenizer(options.pt_dir)
else:
    # Downloading and saving the tokenizer
    tokenizer = get_tokenizer(PARAMETERS['PT_MODEL_PATH'])
    tokenizer.save_pretrained(options.pt_dir)


if __name__ == '__main__':
    with open(options.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if config.train.save and config.train.save_path is not None:
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path)
    if not os.path.exists(config.train.log_dir):
        os.makedirs(config.train.log_dir)

    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)
    config.train.device = device
    config.train.gpus = gpus

    print(config)

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch')

    train_data = TemStaProData(config.train.data_path, options.pt_dir, PARAMETERS['PT_MODEL_PATH'])
    val_data = TemStaProData(config.test.data_path, options.pt_dir, PARAMETERS['PT_MODEL_PATH'])
    print('train size : %d  ||  val size: %d ' % (len(train_data), len(val_data)))
    print('loading data done!')

    model = models.TemStaProModel(PARAMETERS, options)
    optimizer = utils.get_optimizer(config.train.optimizer, model)
    scheduler = utils.get_scheduler(config.train.scheduler, optimizer)

    solver = runner.DefaultRunner(train_data, val_data, model, optimizer, scheduler, tokenizer, gpus, config)
    if config.train.resume_train:
        solver.load(config.train.resume_checkpoint, load_optimizer=True,
                    load_scheduler=True)
    solver.train()


