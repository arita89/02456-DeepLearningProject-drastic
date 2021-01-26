# CLI to evaluate a model

import pandas as pd
import sys
import torch

sys.path.append("../src/evaluation")
sys.path.append("../src/models")
sys.path.append("../src/post-processing")

from eval_dataset import slide_genome, evaluate_all
from analysis import to_confussion_matrix

# Default arguments
expected_args = {
    "--model": None,
    "--genomes": [
        "../data/train-test_genomes/0_0_TEST_GCF_000008525.1_ASM852v1_genomic.fna",
        "../data/train-test_genomes/1_0_TEST_GCF_000008665.1_ASM866v1_genomic.fna",
        "../data/train-test_genomes/2_0_TEST_GCF_000008865.2_ASM886v2_genomic.fna",
        "../data/train-test_genomes/3_0_TEST_GCF_000009045.1_ASM904v1_genomic.fna",
        "../data/train-test_genomes/4_0_TEST_GCF_000027305.1_ASM2730v1_genomic.fna",
        "../data/train-test_genomes/5_0_TEST_GCF_000027325.1_ASM2732v1_genomic.fna",
        "../data/train-test_genomes/6_0_TEST_GCF_000027345.1_ASM2734v1_genomic.fna",
        "../data/train-test_genomes/7_0_TEST_GCF_000091665.1_ASM9166v1_genomic.fna",
        "../data/train-test_genomes/8_0_TEST_GCF_000214725.1_ASM21472v1_genomic.fna",
        "../data/train-test_genomes/9_0_TEST_GCF_003015225.1_ASM301522v1_genomic.fna",
    ],
    "--feat_files": [
        "../data/train-test_genomes/GCA_000008525.1_ASM852v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_000008665.1_ASM866v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_000008865.2_ASM886v2_feature_table.tsv",
        "../data/train-test_genomes/GCA_000009045.1_ASM904v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_000027305.1_ASM2730v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_000027325.1_ASM2732v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_000027345.1_ASM2734v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_000091665.1_ASM9166v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_000214725.1_ASM21472v1_feature_table.tsv",
        "../data/train-test_genomes/GCA_003015225.1_ASM301522v1_feature_table.tsv",
    ],
    "--indeces": [
        (0, 833933),
        (1089200, 2178400),
        (0, 2749289),
        (2107803, 4215606),
        (0, 915069),
        (290038, 580076),
        (0, 408197),
        (832485, 1664970),
        (0, 1273270),
        (1787155, 3574310),
    ],
    "--labels": ["intergenic", "gene"],
    "--each": 100,
    "--window_size": 50,
    "--num_windows": 3,
    "--min_gene_size": 200,
    "--max_gene_size": 2000,
    "--sep": "\t",
    "--out_file": False,
    "--method": "slide_genome",
}


def parse_list_args(arg):
    """ Parse a string list of the form --some_arg=elem1,elem2,elem3 """
    eq_where = arg.find("=")
    if eq_where == -1:
        eq_where = 0
    else:
        return arg[eq_where + 1 :].split(",")


def startswith_dict(key, d):
    """ Check if the string `key` starts any key in the dict `d` """
    for k in d:
        if k.startswith(key):
            return True
    else:
        return False


def loop_args():
    """Parse the arguments, with a --arg=val structure."""
    global expected_args
    args = sys.argv[1:]
    for arg in args:
        pure = arg[0 : arg.find("=")]
        if startswith_dict(pure, expected_args):
            expected_args[pure] = parse_list_args(arg)
            if len(expected_args[pure]) == 0:
                expected_args[pure] = None
            elif len(expected_args[pure]) == 1:
                expected_args[pure] = expected_args[pure][0]
        else:
            print(f"Parameter {arg} not recognized, ignoring that parameter")
    return expected_args


def load_model(model_class, t=50):
    """Instantiate the `model_class` object and load its trained weights."""
    if model_class.lower() == "convlstm":
        from conv_LSTM import convLSTM, criterion

        model = convLSTM(
            input_dim=1, hidden_dim=34, hidden_out=90, output_dim=2, t=t
        )
        path = "../data/models/conv_lstm.pt"

    else:
        raise NotImplementedError(
            "Need to set another step from sequences to indexes using wti.p!"
        )
        path = "../data/models/rnn_embed.pt"
        from embeddings_LSTM import embedLSTM, criterion

        model = embedLSTM()

    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    crit = criterion()

    return model, crit


def apply_evaluation(args):
    """Gather all the functions in the evaluation workflow."""
    # model and method are treated differently (aren't arguments to build test)
    model = args["--model"]
    if args["--method"] != "stored":
        # Evaluation dataset: get all argumetns without the starting "--"
        kwargs = {
            k[2:]: v
            for k, v in args.items()
            if k not in ["--model", "--method"]
        }
        print("Gathering testing dataset...")
        df_in = slide_genome(**kwargs)
    else:
        # generating the whole dataset is quite intensive
        df_in = pd.read_csv("all_testing.tsv", sep="\t")

    # load the trained neural nets
    print("Loading neural models...")
    model, criterion = load_model(model)
    # raw performance
    print("Evaluating performance...")
    valid_acc, rounded_acc, filt_acc, preds, filt_preds, y = evaluate_all(
        model, df_in, criterion
    )
    conf = to_confussion_matrix(y, preds)
    filt_conf = to_confussion_matrix(y, filt_preds)
    # apply lowpass to smooth the signals and enhance the performance
    return valid_acc, rounded_acc, filt_acc, preds, filt_preds, conf, filt_conf


def print_evaluation(final):
    loss, accuracy, accuracy_filter, conf_matrix, conf_matrix_filter = final
    print(
        f"Raw output of Neural Network\n{len('Raw output of Neural Network')*'='}"
        f"\nLoss -> {loss}\nAccuracy -> {accuracy}\nConfusion matrix -> {conf_matrix}"
        f"\n\nLowpass filter\n{len('Lowpass filter')*'='}"
        f"\n{accuracy_filter}\nConfusion matrix -> {conf_matrix_filter}"
    )


if __name__ == "__main__":
    evaluated = apply_evaluation(loop_args())
    print_evaluation(evaluated)
