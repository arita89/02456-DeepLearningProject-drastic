# drastic

Deep Rapid Annotation using Submodels Training in Cells

## Table of contents
<!--ts-->
   * [What is drastic?](#what-is-drastic)
   * [Getting started](#getting-started)
   * [Features](#features)
      * [Further in the horizon](#further-in-the-horizon)
   * [Command Line Interface](#command-line-interface)
   * [Schema of the model](#schema-of-the-model)
   * [What We Learned](#what-we-learned)
   * [Refereces](#references)
<!--te-->

## What is drastic?

Drastic provides a Deep Learning model to perform annotation of a prokaryotic genome.
This project is highly inspired by DeepAnnotator [[1]](#amin2019) (repository [here](https://github.com/ruhulsbu/DeepAnnotator)).

## Getting started

How to clone, add `bin` to the `PATH` and use the model to train a model here.

## Features

This project aims to provide a deep neural network capable of annotating a prokaryotic
genome.

* [ ] Trained LSTM neural network model to find genes in user provided sequences.
* [ ] Embeddings for the NLP problem.
* [ ] Score performance based on [[1]](#amin2019).
* [ ] Trained LSTM neural network model to define start and end of genes in user provided
genome.
* [ ] Trained LSTM neural network to define start and end of genes and also protein coding sequence.
* [ ] Treating of edge-cases (rRNA, tRNA, CRIPRs...).

#### Further in the horizon

Alternatively, the user could provide its own data to train the model.

* [ ] Profiles of sequences.
* [ ] CLI interface.
* [ ] Pre-process data supplied by the user.
* [ ] Train the model with this data.
* [ ] Use the model to predict.
* [ ] [Streamlit](https://streamlit.io/) interface.

## Command Line Interface

Provided as help, in this section, the usage of the model should be summarized.

## Schema of the model

An explanation (better with a graph) should be placed here.

## What We Learned

This project was developed for the MSc course "Deep Learning" of the Technical University of Denmark.

## References

[<a name="amin2019">1</a>] Amin, M. R., Yurovsky, A., Tian, Y., & Skiena, S. (2018). DeepAnnotator. The 2018 ACM International Conference
