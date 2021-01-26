#!/bin/bash

# Script to extract a downloaded genome's feature table into a tsv

mvd
tar -xf genome_assemblies.tar
rm genome_assemblies.tar report.txt
cd ncbi*
gunzip G*gz
genome=`ls | egrep "^G.+txt$"`
mv $genome ../${genome:0:-3}tsv
cd ..
rm -r ncbi*
# Print extracted file to stdout
ls -rt | tail -n1
