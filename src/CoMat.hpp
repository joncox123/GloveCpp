/*
 * cooccurMat.hpp
 *
 *  Created on: Aug 29, 2015
 *      Author: jacox
 */

#ifndef COMAT_HPP_
#define COMAT_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <utility>
#include <unordered_map>
#include <unordered_map>
#include <vector>
#include <armadillo>
#include <cmath>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <tbb/concurrent_hash_map.h>
#include "VocabBuilder.hpp"

using namespace std;
using namespace arma;
using namespace boost;
using namespace algorithm;

struct WordEntry {
	uword i;
	uword j;
	double val;
};

typedef pair<string, unsigned long> VocabPair;
typedef tbb::concurrent_hash_map<string,WordEntry> XMap;

bool valueCompare(VocabPair first, VocabPair second);
double get_current_time_in_ms (void);

class CoMat {
private:

	XMap *xmap;
	VocabMap *vocabMap;
	vector<string> fileVec;

public:
	sp_mat X;

	CoMat();
	CoMat(XMap *xm, VocabMap *vmap, vector<string> fvec) : xmap(xm), vocabMap(vmap), fileVec(fvec) { }
	void operator() ( const tbb::blocked_range<int> &r ) const;
	void buildX(string fname) const;
	void xmapTospMat(XMap *xmap_loc, unique_ptr<VocabMap> const &vmap_loc);
	bool writeX(string fname);
	bool readX(string fname);
	void RandInit(int sz);
};

#endif /* COMAT_HPP_ */
