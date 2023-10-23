import os
import csv
from Bio import SeqIO
import subprocess
import argparse
from time import time

from multiprocessing import Pool


def split_fasta(fasta_file, n, identification):
    dir_path = 'split_fasta' + '_' + identification
    if os.path.exists(dir_path):
        file_names = os.listdir(dir_path)
        for file in file_names:
            os.remove(os.path.join(dir_path, file))
    else:
        os.makedirs(dir_path)
    seqs = list(SeqIO.parse(fasta_file, "fasta"))
    split_size = len(seqs) // n
    output_files = []
    for i in range(n):
        start = i * split_size
        end = (i + 1) * split_size
        file_name = os.path.join(dir_path, f"output_{i}.fasta")
        SeqIO.write(seqs[start:end], file_name, "fasta")
        output_files.append(file_name)
    return output_files


def process_fasta(args):
    input_file, other_params = args
    output_file = f"{os.path.splitext(input_file)[0]}.tsv"
    result = subprocess.run("python iFeature.py --file {} {} --out {}".format(
        input_file, other_params, output_file), shell=True)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="input fasta file", default='/home/zhuxh/data/TemStaPro/val_kickup_small.fasta')
    parser.add_argument("--output_file", help="output tsv file", default='fff.tsv')
    parser.add_argument("--identification", help="identification", default='cksaap')
    parser.add_argument("--other_params", help="params for the worker", default='--type CKSAAP')
    parser.add_argument("--core", help="cpu core number", default=4, type=int)
    parser.add_argument("--merge", help="whether to merge results files", default='True')
    parser.add_argument("--per_image", help="whether to output one by one", default='False')
    parser.add_argument("--clear", help="whether to clear intermediate results files", default='True')
    args = parser.parse_args()

    # Split the input fasta file into 4 smaller fasta files
    start_time = time()
    fasta_files = split_fasta(args.input_file, args.core, args.identification)
    process_args = [(fasta_file,  args.other_params) for fasta_file in fasta_files]
    end_time = time()
    print('splitting completed in {} s'.format(end_time - start_time))

    # Use multiprocessing to process each of the smaller fasta files in parallel
    start_time = time()
    pool = Pool(processes=args.core)
    output_files = pool.map(process_fasta, process_args)
    pool.close()
    pool.join()
    end_time = time()
    print('descriptor completed in {} s'.format(end_time-start_time))

    # Concatenate all output files into a single file
    if args.merge == 'True':
        start_time = time()
        with open(args.output_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for i, tsv_file in enumerate(output_files):
                with open(tsv_file, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    first_row = True
                    for row in reader:
                        if i > 0 and first_row:
                            pass
                        else:
                            writer.writerow(row)
                        first_row = False
        end_time = time()
        print('output-file merge completed in {} s'.format(end_time-start_time))
    elif args.per_image == 'True':
        start_time = time()
        for i, tsv_file in enumerate(output_files):
            with open(tsv_file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                first_row = True
                for row in reader:
                    if i > 0 and first_row:
                        pass
                    else:
                        row
                    first_row = False
        end_time = time()
        print('output-files one by one completed in {} s'.format(end_time-start_time))

    # clear
    if args.clear == 'True':
        print('clear files ...')
        for tsv_file in output_files:
            os.remove(tsv_file)
        for fasta_file in fasta_files:
            os.remove(fasta_file)
