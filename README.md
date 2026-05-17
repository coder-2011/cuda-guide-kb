# CUDA Programming Guide KB

A local knowledge base for you and your agents, all abt cuda! Alot of CUDA information is well documented, but sparsely so, so LLMs don't end up training on the esoteric hard parts. The skill included is built to make agents make tons of queries to the kb so that they never miss information. CUDA isn't as forgiving either, so double checking is very useful. 

There is an agent skill included to use the knowledge base, and PROMPT.md includes a prompt you can copy paste for an agent to set up the knowledge base

I use this alot, I hope you find it useful :)


## Setup Guide

## How It Works

The query CLI uses a local vector index built with scikit-learn TF-IDF vectors:

- Word n-grams help broad CUDA concept queries.
- Character n-grams help exact CUDA names, APIs, and intrinsics.
- Section records help broad questions route into the right part of the guide.
- Query expansion adds CUDA-specific terms for common broad topics like memory, performance, streams, synchronization, and occupancy.

Each result includes the source file, guide page, heading path, score, and excerpt.

Source PDF:

https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf
