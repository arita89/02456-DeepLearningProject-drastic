# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:10:29 2019

@author: mfba
"""

import pandas as pd
import os


def remove_small_genes(annotations, min_gene_size):

    genes_to_remove = []

    # Iterate over rows
    for index, row in annotations.iterrows():

        # Find gene length
        gene_length = row["end"] - row["start"]

        # Check if smaller than minimum length
        if gene_length < min_gene_size:
            genes_to_remove.append(index)

    # Remove all genes smaller than minimum
    annotations = annotations.drop(annotations.index[genes_to_remove])
    return annotations.reset_index(drop=True)


def remove_big_genes(annotations, max_gene_size):

    genes_to_remove = []

    # Iterate over rows
    for index, row in annotations.iterrows():

        # Find gene length
        gene_length = row["end"] - row["start"]

        # Check if smaller than minimum length
        if gene_length >= max_gene_size:
            genes_to_remove.append(index)

    # Remove all genes smaller than minimum
    annotations = annotations.drop(annotations.index[genes_to_remove])
    return annotations.reset_index(drop=True)


if __name__ == "__main__":
    # Local path of git directory
    os.chdir("D:/DTU/02456 - Deep Learning/final_project/drastic/data")

    # Remember to change to local path
    filepath = "GCA_000008865.2_ASM886v2_feature_table.tsv"

    # Save gene meta data in Panda frame
    annotations = pd.read_csv(filepath, sep="\t")

    # Minimum gene size. All genes less than this will be removed.
    min_gene_size = 200

    # Retrieve valid genes
    ann = remove_small_genes(annotations, min_gene_size)

    print("End of script.")
