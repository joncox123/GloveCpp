/*
 * X.cpp
 *
 *  Created on: Aug 29, 2015
 *      Author: jacox
 */

#include "CoMat.hpp"

CoMat::CoMat() {
}

// This is the function called by every iteration of the parallel_for
void CoMat::operator()(const tbb::blocked_range<int> &r) const {
	for (int i = r.begin(); i != r.end(); i++) { // iterates over the entire chunk
		buildX(fileVec[i]);
	}
}

void CoMat::buildX(string fname) const {
	ifstream file;
	char_separator<char> boostCharSep(TOKEN_DELIMETERS);
	unsigned long lineNum = 1;
	uword i, j;
	string line, stmp;
	vector<pair<string, uword>> tokenVec;

	VocabMap::const_accessor va; // Get read access to hashmap
	XMap::accessor xa; // Get write access to hashmap

	if (vocabMap->size() == 0) {
		cerr << "buildX(): vocabMap is empty!" << endl;
		return;
	}

	cout << "Counting " << fname << endl;

	file.open(fname, ios::in);
	if (file.is_open()) {
		//double T_0 = get_current_time_in_ms();
		while (getline(file, line)) {
			tokenVec.clear();
			// Split line into tokens
			tokenizer<char_separator<char>> tokens(line, boostCharSep);
			// Create a list of tokens that are part of our vocabulary
			// Confirmed that iteration is done in order (left to right)
			for (const auto &token : tokens) {
				// Add token into map and increment count
				stmp.assign(token);
				to_lower(stmp);
				// Check if token is part of our vocab
				if (vocabMap->find(va, stmp)) {
					tokenVec.push_back(pair<string,uword>(stmp, va->second));
				}
				va.release();
			}

			// words now contains a map of all words in our vocabulary present on a given line
			// For each word in map, i, increment j for all other words on line
			for (i = 0; i < tokenVec.size(); i++) {
				for (j = 0; j < tokenVec.size(); j++) {
					// Increment the co-occurance count by a scaled amount proportional to the distance
					// of word i from j. if i=j, scaling factor is 1.
					//X(vocabMap[tokenVec[i]], vocabMap[tokenVec[j]]) += 1.0/(double(abs(i-j))+1.0);

					// Since element-wise assignment of a sparse matrix is very slow, we must build up
					// vectors representing non-zero elements, and batch  assign those
					if (i != j) {
						if(xmap->insert(xa, tokenVec[i].first + tokenVec[j].first)) {
							xa->second.i = tokenVec[i].second;
							xa->second.j = tokenVec[j].second;
							xa->second.val = 0;
						}
						xa->second.val += 1.0 / (double(abs(int(i - j)))); // Increment this key's value
						xa.release();
					}
				}
			}

			/*
			if ((lineNum % 10000) == 0) {
				cout << "10,000 lines in " << get_current_time_in_ms() - T_0 << "ms" << endl;
				T_0 = get_current_time_in_ms();
			}
			*/
			++lineNum;

		}
		file.close();
	} else
		cout << "CoMat::buildX, error opening file!";
}

// Convert the hashmap of word pairs to a square sparse matrix
void CoMat::xmapTospMat(XMap *xmap_loc, unique_ptr<VocabMap> const &vmap_loc) {
	if (vmap_loc && (xmap_loc != NULL)) {
		uword V = vmap_loc->size();
		uword len = xmap_loc->size();
		vec Vals = zeros<vec>(len);
		umat IJ = zeros<umat>(2,len);

		uword k = 0;
		XMap::const_iterator it = xmap_loc->begin();
		while ( it != xmap_loc->end() ) {
			Vals(k) = (*it).second.val;
			IJ(0, k) = (*it).second.i;
			IJ(1, k) = (*it).second.j;

			++k;
			++it;
		}

		X = sp_mat(IJ, Vals, V, V);
	}
	else
		cerr << "xmapTospMat(): xmap or vmap are null!" << endl;
}

bool CoMat::writeX(string fname) {
	return !X.save(fname + ".arma");
}

bool CoMat::readX(string fname) {
	return !X.load(fname + ".arma");
}

// Randomly initialize X, for testing purposes
void CoMat::RandInit(int sz) {
	X = sp_mat(sz, sz);
	for (int i = 0; i < sz; i++) {
		for (int j = 0; j < sz; j++) {
			X(i, j) = abs(sin((double) (i + j)));
		}
	}

	//X = randi<mat>(sz, sz, distr_param(0, 3));
	//X = X.t()*X + 1; // make symmetrical
}

double get_current_time_in_ms(void) {
	struct timespec spec;

	clock_gettime(CLOCK_REALTIME, &spec);

	return double(spec.tv_sec) * 1e3 + double(spec.tv_nsec) * 1e-6;
}
