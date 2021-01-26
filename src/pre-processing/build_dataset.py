# Functions for building the dataset that it's going to be used for training
# and testing the Deep Model.

import pandas as pd
import numpy as np


def map_ranges_2_genes(genome, feat_ranges, label):
    """ Map the labeled ranges to the genome sequences, getting a dataframe as 
    output.
    
    Parameters
    ----------
    genome: string
        whole sequence of the DNA
    feat_ranges: np.array
        of 2 len arrays that define label ranges
    label: string
        the actual label

    Returns
    -------
    pandas.DataFrame
        2 columns: sequence and label
    """
    N = len(feat_ranges)
    df = pd.DataFrame(columns=["sequence", "label"])
    df["label"] = [label] * N
    df["sequence"] = [genome[feat[0] : (feat[1] + 1)] for feat in feat_ranges]
    df["start"] = [feat[0] for feat in feat_ranges]
    df["end"] = [feat[1] for feat in feat_ranges]
    return df


def map_full_genomes(
    genome,
    positive,
    negative,
    partial,
    labels=["gene", "intergenic", "partial"],
):
    """ Map the labeled ranges to the genome sequences, getting a dataframe as 
    output.
    
    Parameters
    ----------
    genome: string
        whole sequence of the DNA
    positive: np.array
        of 2 len arrays that define feature ranges
    negatives: np.array
        of 2 len arrays that define no feature ranges
    partial: np.array
        of 2 len arrays that define partial ranges
    labels: list of strings
        the actual labels. Default ["gene", "intergenic", "partial"]

    Returns
    -------
    pandas.DataFrame
        2 columns: sequence and label
    """
    feat = map_ranges_2_genes(genome, positive, labels[0])
    no_feat = map_ranges_2_genes(genome, negative, labels[1])
    partial = map_ranges_2_genes(genome, partial, labels[2])
    return pd.concat([feat, no_feat, partial])


def pad_seq(seq, N, fill="X"):
    """ Extend a str `seq` to a size `N` with character `fill` """
    return seq + fill * (N - len(seq))


def build_padded_seqs(df, N=None):
    """ Given the dataset `df`, pad every sequence to a fixed length
    
    Parameters
    ----------
    df: pandas.DataFrame
        with 2 columns: sequence, label
    N: int
        fixed final size of each sequence. Default to max in `df`
    """
    df_out = df.copy()
    # 1. get length of every sequence
    df_out["seq_length"] = df.iloc[:, 0].apply(len)
    # 2. get max length
    N_data = max(df_out["seq_length"])
    if N is None:
        N = N_data
    elif N < N_data:
        N = N_data
    # 3. transform sequences to fixed-sized padded sequences
    df_out["padded_sequences"] = df_out.iloc[:, 0].apply(
        lambda x: pad_seq(x, N)
    )
    return df_out


def classify_genes(df_ranges, feat_table, feature="gene"):
    """Extract information about name and product accession and add it to the 
    dataframe.
    
    Parameters
    ----------
    df_ranges: pd.DataFrame
        from `map_full_genomes`
    feat_table: pd.DataFrame
        raw, as read from a feature table file from NCBI
    feature: str
        label of interest. Default: "gene", could also be "CDS".
        
    Returns
    -------
    out: pd.DataFrame
        modified `df_ranges`
    """
    global memo

    def get_one_id(st, end, df, ids=["name", "product_accession"]):
        """Vectorize feature -> classifiers.

        Might fail sometimes due to overlapping genes and genes that are larger
        than their corresponding CDS region.
        """

        global memo
        if memo[2] < end:
            # exploit sequentiality of data
            res = df.loc[
                (df.start == st) & (df.end >= end), ids + ["end"]
            ].values
            if res.size == 0:
                return
            else:
                memo = res[0, :]
        return [memo[0], memo[1]]

    df_out = df_ranges.copy()  # remain functional
    out = df_out[df_out.label == feature]
    df = feat_table[feat_table["# feature"] == "CDS"]
    # the informations about name and product is encoded in the CDS; so we
    # need a map from the gene to these identifiers/classes
    memo = [None, None, 0]  # some memoization to speed things up
    names_products = [
        get_one_id(st, end, df) for st, end in zip(out["start"], out["end"])
    ]
    out.loc[:, "name"] = [
        namprod[0] if namprod is not None else None
        for namprod in names_products
    ]
    out.loc[:, "product_accession"] = [
        namprod[1] if namprod is not None else None
        for namprod in names_products
    ]
    df_out["name"] = False
    df_out["product_accession"] = False
    df_out.loc[df_out.label == feature, :] = out
    return df_out


if __name__ == "__main__":
    # TODO: build better tests with actual genomic data
    ran = np.array([[4, 8], [12, 27], [34, 39], [41, 47]], dtype="int32")
    seq = "CFGYUJMNBIUGVOIJBUIGFUYHBLKJBHVJYTDFIUYGUYVUGUYDTRDFIYFYTIFIYTFYFIYTFIYITYFIYTFHVBVGY"
    label = "gene"
    df = map_ranges_2_genes(seq, ran, label)
    print(build_padded_seqs(df))
