/*
 * Train.cpp
 *
 *  Created on: Aug 29, 2015
 *      Author: jacox
 */

#include "Train.hpp"

#include "CoMat.hpp"

bool Train::setup(CoMat *X, unsigned int dims, double alpha, double xmax, bool vecCost) {
	this->X = X;
	this->Wdims = dims;
	this->alpha = alpha;
	this->xmax = xmax;
	this->Nv = X->X.n_rows;

	// Randomly initialize word vectors
	Wi = 0.25 * randn(Wdims, Nv);
	Wj = 0.25 * randn(Wdims, Nv);

	// Initialize bias vectors
	bi = 0.25 * randn<vec>(Nv);
	bj = 0.25 * randn<vec>(Nv);

	dJdWi = zeros<mat>(Wi.n_rows, Wi.n_cols);
	dJdWj = zeros<mat>(Wj.n_rows, Wj.n_cols);
	dJdBi = zeros<vec>(bi.n_elem);
	dJdBj = zeros<vec>(bj.n_elem);

	// For vectorized cost function, divide sparse matrix into sub-matricies of size BLOCK_SZ
	if (vecCost) {
		blockSz = bruteFactor(uword(Nv), uword(Nv), uword(Nv/10));
		if (isnan(blockSz)) {
			cerr << "Failed to find suitable sub-multiple for vocabulary size of " << Nv <<
					". Please chose a a vocab size (-s) which is, preferably, a multiple of 1000, 10000, 20000, etc." << endl;
			return true;
		}
		else
			cout << "Sub-matrix block size set to: " << blockSz << endl;
	}

	return false;
}

void Train::sp_log(sp_mat &X) {
	for(sp_mat::iterator x_ij=X.begin(); x_ij!=X.end(); ++x_ij) {
		*x_ij = log(1.0 + double(*x_ij));
	}
}

// Apply squashing function to Co-occurance matrix
void Train::F(sp_mat &X) {
	// Unfortunately, arma doesn't support many sparse matrix operations
	// This includes pow and find. Therefore, we have to iterate through
	// the matrix to perform the commented operation:
	for(sp_mat::iterator x_ij=X.begin(); x_ij!=X.end(); ++x_ij) {
		*x_ij = pow(double(*x_ij / xmax), alpha);
		if (*x_ij > xmax)
			*x_ij = xmax;
	}
}

mat * Train::F(mat *Xm) {
	*Xm = arma::pow(*Xm / xmax, alpha);
	Xm->elem(find(*Xm >= xmax)).ones();

	return Xm;
}

double Train::F(double x) {
	double y = pow(x/xmax, alpha);

	return (y > xmax) ? xmax : y;
}

// Slow looping cost function
double Train::CostLoop() {
	double J = 0;
	double Aij, xf;
	double Vinv = 1.0/Nv;

	dJdWi *= 0;
	dJdWj *= 0;
	dJdBi *= 0;
	dJdBj *= 0;

	uword i,j;
	for(sp_mat::const_iterator e=X->X.begin(); e!=X->X.end(); e++) {
		i = e.row();
		j = e.col();
			Aij = as_scalar(Wi.col(i).t() * Wj.col(j) + bi(i) + bj(j)) - std::log(double(X->X(i, j))+1.0);
			xf = F(double(X->X(i, j)));
			J += xf * std::pow(Aij, 2);

			// Compute gradients
			// Each dimension of a given word vector i is Aij * sum(Wj(:,j))
			// In other words, each dimension of word vector i in Wi is proportional to the sum of word vector j
			for (unsigned int k = 0; k < Wdims; k++) {
				dJdWi(k, i) += xf * Aij * Wj(k, j);
				dJdWj(k, j) += xf * Aij * Wi(k, i);
			}

			dJdBi(i) += xf* Aij;
			dJdBj(j) += xf * Aij;
		}

	dJdWi *= Vinv;
	dJdWj *= Vinv;
	dJdBi *= Vinv;
	dJdBj *= Vinv;
	return 0.5 * Vinv * J;
}

// Fast vectorized Cost function
double Train::Cost() {
	mat A, AXfsm, Xsm;
	double J = 0;
	double Vinv = 1.0/Nv;
	uword wi0, wif, wj0, wjf;
	uword Nblocks;

	dJdWi *= 0;
	dJdWj *= 0;
	dJdBi *= 0;
	dJdBj *= 0;

	Nblocks = uword(ceil(Nv/double(blockSz)));

	// Break computation in sub-matrix blocks of 10,000 x 10,000 word vectors.
	for (uword i=0; i<Nblocks; i++) {
		for(uword j=0; j<Nblocks; j++) {
		// Determine sub-matrix indicies for this iteration
		wi0 = i*blockSz;
		wj0 = j*blockSz;
		if (i == Nblocks-1)
			wif = Nv-1;
		else
			wif = (i+1)*blockSz-1;
		if (j == Nblocks-1)
			wjf = Nv-1;
		else
			wjf = (j+1)*blockSz-1;

		// Extract sub-matrix view
		Xsm = X->X.submat(wi0, wj0, wif, wjf);

		// Main calculation of cost and grad
		A = (Wi.cols(wi0, wif)).t() * Wj.cols(wj0, wjf) - log(Xsm + 1.0);

		A.each_col() +=  bi.rows(wi0,wif); // add in bias
		A.each_row() += (bj.rows(wj0,wjf)).t(); // add in bias

		// Extract submatrixes
		AXfsm = A % (*(F(&Xsm))); // NOTE: Xsm has been transformed in place to Xf!

		// here, % is element-wise multiplication, not modulo
		//J = .5 * Vinv * sum(sum(Xf % pow(A, 2)));
		J += .5 * Vinv * sum(sum(AXfsm % A));

		dJdWi.cols(wi0, wif) += Vinv * (AXfsm * (Wj.cols(wj0, wjf)).t()).t();
		dJdWj.cols(wj0, wjf) += Vinv * Wi.cols(wi0, wif) * AXfsm;

		dJdBi.rows(wi0,wif) += Vinv * mat(sum(AXfsm, 1));
		dJdBj.rows(wj0,wjf) += Vinv * mat(sum(AXfsm, 0).t());
		}
	}
	return J;
}

double Train::GradDesc(unsigned int max_iters, double eta, double p, bool vecCost) {
	double J = 0;
	double eta_t = eta;
	double T_0, dT; // Store time required to complete an iteration

	// Store weight updates for momentum (prefix p for momentum is from physics convention)
	mat dWi = zeros(Wi.n_rows, Wi.n_cols);
	mat dWj = zeros(Wi.n_rows, Wi.n_cols);
	vec dBi = zeros(bi.n_elem);
	vec dBj = zeros(bj.n_elem);

	cout << "Training with gradient descent" << endl;
	for (unsigned int i = 0; i < max_iters; i++) {
		T_0 = get_current_time_in_ms();
		// Only use potentially faster, but much more memory intensive, vectorized cost function if specified
		if (vecCost) {
			J = Cost();
		}
		else
			J = CostLoop();
		dT = get_current_time_in_ms() - T_0;

		dWi = eta_t * dJdWi + p * dWi;
		dWj = eta_t * dJdWj + p * dWj;
		dBi = eta_t * dJdBi + p * dBi;
		dBj = eta_t * dJdBj + p * dBj;

		Wi -= dWi;
		Wj -= dWj;
		bi -= dBi;
		bj -= dBj;

		// Decay the learning rate according to a schedule
		eta_t = eta_t * (1.0 - 1.5 / ((double) max_iters));

		cout << "Cost: " << J << ", on iteration " << i << " with learning rate " << eta_t  << ", in " << dT << " ms." << endl;
	}

	return J;
}

bool Train::SaveWordVecs(string fname) {
	bool suc1, suc2, suc3, suc4;
	suc1 = bi.save(fname + "_bi.csv", csv_ascii);
	suc2 = bj.save(fname + "_bj.csv", csv_ascii);

	suc3 = Wi.save(fname + "_Wi.csv", csv_ascii);
	suc4 = Wj.save(fname + "_Wj.csv", csv_ascii);

	return !(suc1 && suc2 && suc3 && suc4);
}

bool Train::CheckGrad(bool vecCost) {
	const int Nbench = 100;
	const double h = 1e-4; // finite difference step size
	double J1, J2, J;
	mat dJdWi_num, dJdWj_num, dJdBi_num, dJdBj_num;
	double (Train::*costFunc)();

	if (vecCost)
		costFunc = &Train::Cost;
	else
		costFunc = &Train::CostLoop;

	// Initialize numerical gradients to zero
	dJdWi_num = zeros<mat>(Wi.n_rows, Wi.n_cols);
	dJdWj_num = zeros<mat>(Wj.n_rows, Wj.n_cols);
	dJdBi_num = zeros<vec>(bi.n_elem);
	dJdBj_num = zeros<vec>(bj.n_elem);

	// Run a benchmark
	double T_0, dT; // Store time required to complete an iteration
	T_0 = get_current_time_in_ms();
	for (int z=0; z<Nbench; z++) {
		(*this.*costFunc)();
	}
	dT = get_current_time_in_ms() - T_0;
	cout << "Benchmark, time per iteration: " << dT/double(Nbench) << " ms." << endl;

	// Evaluate numerical gradient of Ws
	for (unsigned int a = 0; a < dJdWi.n_rows; a++) {
		for (unsigned int b = 0; b < dJdWi.n_cols; b++) {
			/****** Wi ********/
			Wi(a, b) -= h;
			J1 = (*this.*costFunc)();

			Wi(a, b) += 2 * h;
			J2 = (*this.*costFunc)();

			// Compute center difference
			dJdWi_num(a, b) = (J2 - J1) / (2 * h);

			// Restore original values
			Wi(a, b) -= h;

			/****** Wj ********/
			Wj(a, b) -= h;
			J1 = (*this.*costFunc)();

			Wj(a, b) += 2 * h;
			J2 = (*this.*costFunc)();

			// Compute center difference
			dJdWj_num(a, b) = (J2 - J1) / (2 * h);

			// Restore original values
			Wj(a, b) -= h;
		}
	}

	// Evaluate numerical gradient of biases
	for (unsigned int a = 0; a < dJdBi.n_elem; a++) {
		/****** bi ********/
		bi(a) -= h;
		J1 = (*this.*costFunc)();

		bi(a) += 2 * h;
		J2 = (*this.*costFunc)();

		// Compute center difference
		dJdBi_num(a) = (J2 - J1) / (2 * h);

		// Restore original values
		bi(a) -= h;

		/****** bj ********/
		bj(a) -= h;
		J1 = (*this.*costFunc)();

		bj(a) += 2 * h;
		J2 = (*this.*costFunc)();

		// Compute center difference
		dJdBj_num(a) = (J2 - J1) / (2 * h);

		// Restore original values
		bj(a) -= h;
	}

	// Evaluate "analytic" gradient
	J = (*this.*costFunc)();

	double errWi = sum(sum(arma::abs(dJdWi_num - dJdWi)));
	double errWj = sum(sum(arma::abs(dJdWj_num - dJdWj)));
	double errBi = sum(sum(arma::abs(dJdBi_num - dJdBi)));
	double errBj = sum(sum(arma::abs(dJdBj_num - dJdBj)));
	double errTotal = errWi + errWj + errBi + errBj;
	cout << "J = " << J << ", errWi = " << errWi << ", errWj = " << errWj
			<< ", " << "errBi = " << errBi << ", errBj = " << errBj << endl;

	return (errTotal > 1e-8) ? true : false;
}

bool sortcompdesc (uword i, uword j) {
        return (i > j);
}

// Find a suitable divisor/factor for an integer
uword bruteFactor(uword N, const uword MAX_FACTOR_SZ, const uword MIN_FACTOR_SZ) {
	vector<uword> factors;
	uword Nfact = 1;
	uword chosenFact = numeric_limits<uword>::quiet_NaN();

	while (Nfact <= N) {
		// Does it evenly divide N?
		if (!(N % Nfact))
			factors.push_back(Nfact);
		Nfact++;
	}

	sort(factors.begin(), factors.end(), sortcompdesc);
	// Find largest suitable factor
	for (auto &f : factors) {
		if ((f >= MIN_FACTOR_SZ) && (f <= MAX_FACTOR_SZ)) {
			chosenFact = f;
			break;
		}
	}

	return chosenFact;
}
