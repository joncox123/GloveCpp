/*
 * Train.hpp
 *
 *  Created on: Aug 29, 2015
 *      Author: jacox
 */

#ifndef TRAIN_HPP_
#define TRAIN_HPP_

#include <armadillo>
#include <time.h>
#include <random>
#include <algorithm>
#include <vector>
#include <cmath>

#include "CoMat.hpp"

bool sortcompdesc (uword i, uword j);
uword bruteFactor(uword N, const uword MAX_FACTOR_SZ, const uword MIN_FACTOR_SZ);

class Train {
private:
	mat Wi, Wj, dJdWi, dJdWj; // word vectors, co-occurance word vectors
	CoMat *X;
	vec bi, bj, dJdBi, dJdBj;
	unsigned int Wdims; // dimensionality of word vectors
	double alpha, xmax;
	double Nv; // Vocabulary size (dimensions of square co-occurance matrix, X;
	uword blockSz;

public:
	bool setup(CoMat *X, unsigned int dims, double alpha, double xmax, bool vecCost);
	void F(sp_mat &X);
	mat * F(mat *Xm);
	double F(double x);
	void sp_log(sp_mat &X);
	double Cost();
	double CostLoop();
	double GradDesc(unsigned int max_iters, double eta, double p, bool vecCost = false);
	bool CheckGrad(bool vecCost = false);
	bool SaveWordVecs(string fname);
};

#endif /* TRAIN_HPP_ */
