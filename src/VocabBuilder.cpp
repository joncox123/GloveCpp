/*
 * VocabBuilder.cpp
 *
 *  Created on: Sep 11, 2015
 *      Author: jacox
 */

#include "VocabBuilder.hpp"

VocabBuilder::VocabBuilder(VocabMap *vmap) {
	this->vocabMap = vmap;
}

VocabBuilder::VocabBuilder() {
}

bool valueCompare(VocabPair wordOne, VocabPair wordTwo) {
	return wordOne.second > wordTwo.second;
}

// This is the function called by every iteration of the parallel_for
void VocabBuilder::operator() ( const tbb::blocked_range<int> &r ) const {
	for ( int i = r.begin(); i != r.end(); i++ ) { // iterates over the entire chunk
		buildVocabMap(fileVec[i]);
	}
}

void VocabBuilder::buildVocabMap(string fname) const {
	boost::char_separator<char> boostCharSep = boost::char_separator<char>(TOKEN_DELIMETERS);
	VocabMap::accessor a; // Get writable access to hashmap
		ifstream file;

		file.open(fname, ios::in);
		string line, tmp;
		if (file.is_open()) {
			cout << "Processing file " << fname << endl;
			while (getline(file, line)) {
				// Split line into tokens
				boost::tokenizer<boost::char_separator<char>> tokens(line, boostCharSep);
				// Iterate over tokens
				for (const auto &token : tokens) {
					// Add token into map and increment count
					tmp.assign(token);
					// Only consider words with two or more characters
					if (tmp.length() > 1) {
						boost::to_lower(tmp);
						vocabMap->insert(a, tmp); // Put this key in the map
						a->second += 1; // Increment this key's value
						a.release();
					}
				}
			}
			file.close();
		} else
			cout << "cooccurMat::buildVocab, error opening file!";
}

bool VocabBuilder::saveVocabMap(string vocabCSVfile, unique_ptr<VocabVector> const &vvec) {
	ofstream csvFile;
	csvFile.open(vocabCSVfile, ios::out | ios::trunc);
	if (csvFile.is_open()) {
		cout << "Writing vocabulary list to " << vocabCSVfile << endl;
		for (auto &word : *vvec) {
			csvFile << word.first << "," << word.second << endl;
		}
		csvFile.close();
	} else {
		cout << "cooccurMat::saveVocabMap; couldn't open output file!";
		return true;
	}

	return false;
}

unique_ptr<VocabVector> VocabBuilder::pairVocabMap(unsigned long vocabSize) {
	// Dump hashmap into vector, then sort vector by value
	VocabVector vvec(vocabMap->begin(), vocabMap->end());
	sort(vvec.begin(), vvec.end(), &valueCompare);


	return unique_ptr<VocabVector>(new VocabVector(vvec.begin(), vvec.begin()+vocabSize));
}

unique_ptr<VocabMap> VocabBuilder::readVocabCSV(string fname) {
	unique_ptr<VocabMap> vmap(new VocabMap);
	VocabMap::accessor a; // Get writable access to hashmap

	ifstream file;

	file.open(fname, ios::in);
	string line, tmp;
	if (file.is_open()) {
		unsigned int i = 0;
		cout << "Reading " << fname << endl;
		while (getline(file, line)) {
			// Break line into comma separated tokens
			boost::tokenizer<boost::char_separator<char>> tokens(line);
			const auto token = tokens.begin(); // get first token

			vmap->insert(a, string(*token)); // Put this key in the map
			a->second = i; // Set value for key
			a.release();

			i++;
		}
		file.close();
	} else
		cerr << "cooccurMat::ReadVocabCSV, error opening file!";

	return vmap;
}


