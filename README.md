### GloveCpp

### Introduction
GloVe C++ is an implementation of the GloVe algorithm for learning word vectors from a corpus. The details of this algorithm are described by Pennington, Socher and Manning:

Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "Glove: Global vectors for word representation." Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014) 12 (2014): 1532-1543.

This implementation takes a modern, C++ approach. In particular, it leverages the Intel Thread Building Blocks (TBB) library for efficient parallelization when building a vocabulary and the co-occurrence matrix from a corpus. The TBB library also provides an efficient, thread safe hashmap for storing this data. In addition, GloVe C++ provides a vectorized cost function which uses the sparse matrix features in the Armadillo library to parallelize calculation of the gradient and cost/lost during gradient descent. 
While this approach may or may not be faster than an efficient, multi-threaded implementation in C, it is nevertheless simpler to read, understand and modify. A vectorized implementation is also potentially capable of greater performance with a suitable matrix library, by exploiting modern CPU architectures.

### Feature Requests or Bugs
If you have a feature request, or if you find a bug, you can contact the author at either: jacox@sandia.gov or joncox@alum.mit.edu.
 
### Compiling the Source Code
To compile GloVe, you will need the following libraries and compiler installed on your system:
•	Armadillo 5.4 or greater
•	Intel Thread Building Blocks 4.2 or greater
•	Boost 1.55 or greater
•	G++ compiler 4.9 or greater.
Simply run the make command from within the Release directory to compile GloVe.
 
### Operation
Using GloVe C++ is divided into three stages:
1.	Building a vocabulary from a corpus of text. 
2.	Building the co-occurrence matrix from a corpus of text.
3.	Learning the word vectors from the co-occurrence matrix.
