/*
 * VocabBuilder.hpp
 *
 *  Created on: Sep 11, 2015
 *      Author: jacox
 */

#ifndef VOCABBUILDER_HPP_
#define VOCABBUILDER_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/blocked_range.h>

using namespace std;

#define TOKEN_DELIMETERS ",.'\"@:!?$-_/%+&*;<>=()1234567890 "

typedef pair<string, unsigned long> VocabPair;
typedef tbb::concurrent_hash_map<string,unsigned long> VocabMap;
typedef vector<VocabPair> VocabVector;

class VocabBuilder {
private:
	VocabMap *vocabMap;
	vector<string> fileVec;

public:
	// Functor for parallel_for to pass arguments to operator
	VocabBuilder(VocabMap *vmap, vector<string> fvec) : vocabMap(vmap), fileVec(fvec) { }
	VocabBuilder(VocabMap *vmap);
	VocabBuilder();
	void operator() ( const tbb::blocked_range<int> &r ) const;
	void buildVocabMap(string fname) const;
	unique_ptr<VocabVector> pairVocabMap(unsigned long vocabSize);
	bool saveVocabMap(string vocabCSVfile, unique_ptr<VocabVector> const &vvec);
	unique_ptr<VocabMap> readVocabCSV(string fname);
};

#endif /* VOCABBUILDER_HPP_ */
