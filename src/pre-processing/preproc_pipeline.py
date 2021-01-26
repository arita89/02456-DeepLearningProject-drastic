"""Join all the pre-processing routines into a pipeline."""

import pandas as pd
import pyfastx
import re

from build_dataset import build_padded_seqs, map_full_genomes, classify_genes
from get_labeled_genes import (
    get_feature_ranges,
    tweak_ranges,
    get_negative_ranges,
    sample_negatives,
    get_partial_ranges,
    extract_windows,
    filter_ranges,
)


def remove_unk(df, seq_col="sequence", known=None):
    """Remove sequences with unknown characters."""
    if not known:
        # default to DNA
        known = {"A", "C", "G", "T"}
    # under the hood, pandas is calling re.search, so the pattern is compiled
    # here to speed up the process
    pat_known = re.compile(rf'[^{"".join(known)}]')
    df_filtered = df[~df[seq_col].str.contains(pat_known)]
    return df_filtered


def pre_process(
    genome_file,
    feat_file,
    min_gene_size=200,
    max_gene_size=2000,
    sep="\t",
    out_file=False,
):
    """Master function.

    Accepts a `genome_file` FASTA and a `feat_file` TSV and convers it to a pandas
    dataframe.

    Parameters
    ----------
    genome_file: string
        path to FASTA file
    feat_file: string
        path to corresponding NCBI TSV feature file
        TODO: specify how to get this file from NCBI
    min_gene_size: int
        minimum gene size accepted as input. Defautl to 200
    sep: string
        sep argument to the pandas's read and write functions. Default to "\t"
    out_file: False || string
        if a string is passed, write the dataframe to a file `string`. Default to False

    Returns
    -------
    pandas.DataFrame
        final dataframe to use in training/testing

    """
    # 0. CONSISTENCY OF INPUT
    min_gene_size = int(min_gene_size)

    # 1. READ FILES
    # We expect to have the whole genome as the fist sequence in the multiFASTA
    # file and then some plasmids.
    # TODO: check if that works for a 1-seq FASTA file
    fna = pyfastx.Fasta(genome_file)
    genome = str(fna[0])
    del fna
    df_feat = pd.read_csv(feat_file, sep=sep)
    genome_length = len(genome)

    # 2. STANDARIZE DATA
    df_feat = df_feat[
        df_feat.apply(
            lambda x: x.end - x.start < max_gene_size
            and x.end - x.start > min_gene_size,
            axis=1,
        )
    ]

    # 3. GET RANGES OF FEATURES
    # Test genes
    gene_ranges = get_feature_ranges(df_feat)
    positive = tweak_ranges(gene_ranges, genome_length=genome_length)
    # Test negative examples
    negative_ranges = get_negative_ranges(gene_ranges)
    negative = sample_negatives(negative_ranges)
    # Test partial genes
    partial = get_partial_ranges(gene_ranges, genome_length=genome_length)

    # 4. BUILD THE PRE-PROCESSED DATAFRAME
    df_out = map_full_genomes(genome, positive, negative, partial)
    df_out = df_out[df_out.sequence.apply(lambda x: len(x)) < max_gene_size]
    df_out = build_padded_seqs(df_out)

    if out_file:
        df_out.to_csv(out_file, sep=sep, index=False)
    return df_out


def window_pipeline(
    genome_file,
    feat_file,
    window_size=50,
    num_windows=3,
    min_gene_size=200,
    max_gene_size=2000,
    sep="\t",
    out_file=False,
    by=None,
    manage_unk=True,
    annotate=True,
):
    """Preprocessing pipeline that take equidistant windows in `feat_file`.

    Parameters
    ----------
    (...)
    by: tuple (length 2)
        the first element corresponds to the first position in `genome_file`
        with respect to the real genome in `feat_file`. Only used when portions
        of a genome are provided.
    manage_unk: bool
        whether sequences with unknown characters - not one of A,G,C,T - should
        be removed. These characters (N, S, etc.) are produced when, for
        instanced, the experimental or assembly methods could not resolve a
        position; i.e., there is uncertainty for the identity of this position.
        If False, the sequences with these characters are kept. Default `True`.
    annotate: bool
        whether the fixed sequences that correspond to portions of genes should
        be annotated with columns in the final dataset that provides
        information about the gene function. Default `True`.

    Returns
    -------
    pandas.DataFrame
        final dataframe to use in training/testing

    """
    # 0. CONSISTENCY OF INPUT
    window_size = int(window_size)
    num_windows = int(num_windows)

    # 1. READ FILES
    # We expect to have the whole genome as the fist sequence in the multiFASTA
    # file and then some plasmids.
    # TODO: check if that works for a 1-seq FASTA file
    fna = pyfastx.Fasta(genome_file)
    genome = str(fna[0])
    # just avoid segmentation problems with the lower level API
    del fna
    df_feat = pd.read_csv(feat_file, sep=sep)
    genome_length = len(genome)

    # 2. STANDARIZE DATA
    df_feat = df_feat[
        df_feat.apply(
            lambda x: x.end - x.start < max_gene_size
            and x.end - x.start > min_gene_size,
            axis=1,
        )
    ]

    # 3. GET RANGES OF FEATURES
    # Test genes
    gene_ranges = get_feature_ranges(df_feat)
    if by is not None:
        gene_ranges = filter_ranges(gene_ranges, first=by[0], last=by[1])
    positive = extract_windows(gene_ranges, window_size, num_windows)
    # Test negative examples
    negative_ranges = get_negative_ranges(gene_ranges)
    negative = extract_windows(negative_ranges, window_size, num_windows)
    # Test partial genes
    partial_ranges = get_partial_ranges(
        gene_ranges, genome_length=genome_length
    )
    partial = extract_windows(partial_ranges, window_size, num_windows)

    # 4. BUILD THE PRE-PROCESSED DATAFRAME
    df_out = map_full_genomes(genome, positive, negative, partial)
    if manage_unk:
        df_out = remove_unk(df_out)
    if annotate:
        df_out = classify_genes(df_out, df_feat, feature="gene")

    return df_out


if __name__ == "__main__":
    # Test with Escherichia coli's genome and feature table
    genome = "../../data/GCF_000008865.2_ASM886v2_genomic.fna"
    feature_table = "../../data/GCA_000008865.2_ASM886v2_feature_table.tsv"
    df = window_pipeline(genome, feature_table)
    print(
        f"\nRESULT OF TEST\n{len('RESULT OF TEST')*'='}\n{df.head(10)}\n"
        f"\ncolums -> {list(df.columns)}"
        f"\nlabels -> {pd.unique(df.label)}"
        f"\nNumber of genes -> {(df.label=='gene').sum()}"
        f"\nNon indentified genes ranges -> {df.loc[df.label=='gene','name'].isna().sum()}"
    )
    assert not df.sequence.apply(lambda x: len(x) != 50).any()
