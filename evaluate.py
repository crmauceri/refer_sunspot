__author__ = 'mauceri'

"""
Processes REFER objects and csv files into the correct list format for NLGEval
NLGEval generates machine translation metrics such as Bleu, ROUGE_L, CIDEr, and METEOR for referring expressions
"""

from tqdm import tqdm
from refer import REFER
import numpy as np
from csv import DictReader

from nlgeval import NLGEval

# Loads and executes the models
# Metrics that are calculated by default are Bleu, ROUGE_L, CIDEr, and METEOR
def evaluate(hypothesis, references, no_skipthoughts=True, no_glove=True, metrics_to_omit=[]):
    nlgeval = NLGEval(no_skipthoughts=no_skipthoughts, no_glove=no_glove, metrics_to_omit=metrics_to_omit)
    return nlgeval.compute_metrics(references, hypothesis)

# Holds out one expression from each set of expressions in refer and evaluates as if that expression were the hypothesis and the others were references
# Provides a measure of the varience of the human annotators of the dataset
def self_evaluate(refer, no_skipthoughts=True, no_glove=True, metrics_to_omit=[]):
    hypothesis = []
    references = []

    for ref_id, ref in refer.Refs.items():
        if len(ref['sentences'])>1:
            hypothesis.append(ref['sentences'][0]['sent'])
            references.append([s['sent'] for s in ref['sentences'][1:]])

    return evaluate(hypothesis, references, no_skipthoughts, no_glove, metrics_to_omit)

# Load the hypothesis from a csvfile
# CSV file must have the columns "refID" and "generated_sentence"
def csv_evaluate(csvpath, refer, no_skipthoughts=True, no_glove=True, metrics_to_omit=[]):
    hypothesis = []
    references = []

    with open(csvpath, 'r') as csvfile:
        genData = DictReader(csvfile)
        for row in genData:
            ref_id = int(row['refID'])
            gen_sentence = row['generated_sentence'].replace('<eos>', '')
            hypothesis.append(gen_sentence)
            references.append([s['sent'] for s in refer.Refs[ref_id]['sentences']])

    return evaluate(hypothesis, references, no_skipthoughts, no_glove, metrics_to_omit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('mode', help='self/csv')
    parser.add_argument('csvpath', help='Filepath to csv file with hypothesis', default='')

    parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
    parser.add_argument('--dataset', help='dataset name', default='sunspot')
    parser.add_argument('--splitBy', help='team that made the dataset splits', default='boulder')

    args = parser.parse_args()

    refer = REFER(dataset=args.dataset, splitBy=args.splitBy, data_root=args.data_root)

    if args.mode == "self":
        metrics_dict = self_evaluate(refer)
    elif args.mode == "csv":
        metrics_dict = csv_evaluate(args.csvpath, refer)
    else:
        print('Unsupported mode')
        exit()

    print("Bleu: %3.3f" % metrics_dict['Bleu_1'])
    print("ROUGE_L: %3.3f" % metrics_dict['ROUGE_L'])
    print("CIDEr: %3.3f" % metrics_dict['CIDEr'])
