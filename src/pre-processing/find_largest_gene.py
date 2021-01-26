# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:51:35 2019

@author: mfba
"""

import pandas as pd 
import os 

def find_largest_gene(annotations):
    
    largest_gene = 0
    
    # Iterate over rows
    for index, row in annotations.iterrows():
        
        # Find gene length
        gene_length = row['end']-row['start']
        
        # Check if smaller than minimum length
        if gene_length > largest_gene:
            largest_gene = gene_length
            
    return largest_gene


if __name__ == '__main__':
    # Local path of git directory
    os.chdir('D:/DTU/02456 - Deep Learning/final_project/drastic/data')
    
    # Remember to change to local path
    filepath = "GCA_000008865.2_ASM886v2_feature_table.tsv"
    
    # Save gene meta data in Panda frame
    annotations = pd.read_csv(filepath,  sep='\t')
    
    # Retrieve valid genes
    largest_gene = find_largest_gene(annotations)
    
    print('Largest gene: ' + str(largest_gene))
    
    print('End of script.')