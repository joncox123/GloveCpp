//============================================================================
// Name        : GloVe.cpp
// Author      : Jonathan Cox, joncox@alum.mit.edu
// Version     :
// Copyright   : Sandia National Laboratories, 2015
// Description : Sparse matrix implementation of GloVe in C++11. Requires Armadillo 5.5 or greater.
//============================================================================

#include <iostream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <memory>
#include <vector>
#include <tclap/CmdLine.h>
#include <boost/filesystem.hpp>
#include <iterator>
#include <tbb/concurrent_hash_map.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "CoMat.hpp"
#include "Train.hpp"
#include "VocabBuilder.hpp"

using namespace std;
using namespace arma;
using namespace boost::filesystem;

#define VERSION 1.0
#define GLOVE_DESCRIPTION "Learn word vectors from a corpus using GloVe algorithm described by Pennington, Socher and Manning. "\
		"Includes vectorized cost function using Armadillo sparse matrix library.\n"\
		"Author: Jonathan A. Cox <joncox@alum.mit.edu>."

#define LICENSE_TEXT "------------------------------------------------------------------------"\
	 "Approved for Unclassified Unlimited Release (UUR), SAND No: SAND2015-8080 O.\n"\
	 "Copyright (c) 2015 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 "\
         "with Sandia Corporation, the U.S. Government retains certain rights in this software. "\
         "Redistribution and use in source and binary forms, with or without modification, "\
         "are permitted provided that the following conditions are met:\n"\
             "Redistributions of source code must retain the above copyright notice, this list "\
             "of conditions and the following disclaimer. Redistributions in binary form must "\
             "reproduce the above copyright notice, this list of conditions and the following "\
             "disclaimer in the documentation and/or other materials provided with the distribution. "\
             "Neither the name of the Sandia Corporation nor the names of its contributors may "\
             "be used to endorse or promote products derived from this software without specific "\
             "prior written permission.\n"\
         "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY "\
         "EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES "\
         "OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT "\
         "SHALL SANDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, "\
         "EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF "\
         "SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS "\
         "INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, "\
         "STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY "\
         "OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."


#define DEF_CORPUS_PATH "corpus"
#define DEF_NUM_FILES 99
#define DEF_VOCAB_LIST_FNAME "vocabList.csv"
#define DEF_VOCAB_SIZE 10000
#define DEF_WORD_VEC_FNAME "wvec"
#define DEF_X_FNAME "X"
#define DEF_WORD_VEC_DIMS 100
#define DEF_ALPHA 0.75
#define DEF_XMAX 100
#define DEF_ETA 0.025 // learning rate
#define DEF_MAX_ITERS 100
#define DEF_MOMENTUM 0.9
#define DEF_NUM_THREADS 1
#define DEF_VEC_COST true

enum RunMode {
	BUILD_VOCAB, BUILD_X, TRAIN, CHECK_GRAD
};

struct RunParameters {
	string corpusPath;
	string vocabListFileName;
	string wordVecsFileName;
	string XfileName;
	int vocabSize;
	int wordVecDims;
	double alpha;
	int xmax;
	double eta;
	int maxIters;
	double momentum;
	int numThreads;
	bool vecCost;
};

// Return false if value doesn't lie within range (not inclusive)
template<typename T>
bool rangeCheck(const T &value, const T low, const T high) {
	return !((low < value) && (value < high));
}

// Local function prototypes
vector<string> readDirectory(string path);
bool processCommandlineArguments(RunParameters *runParams, RunMode *runMode, int argc, char **argv);

// Command line arguments: <begining file number> <ending file number> <run_identifier>
int main(int argc, char **argv) {
	RunParameters runParams;
	RunMode runMode;
	bool err;
	CoMat coMat; // Reduced cooccurance matrix
	Train trainer;
	VocabMap *vmap = new VocabMap();
	vector<string> fileVec;
	VocabBuilder vocabBuilder(vmap);
	unique_ptr<VocabVector> vvec(new VocabVector);
	unique_ptr<VocabMap> vmap_up;
	XMap *xmap = new XMap();

	if (processCommandlineArguments(&runParams, &runMode, argc, argv))
		exit(EXIT_FAILURE);

	cout << "GloVe C++ (Armadillo, Intel Thread Building Blocks) v" << setprecision(2) << VERSION << endl;

	// Proceed to perform actual word vector training or algorithm unit test
	switch (runMode) {
	case BUILD_VOCAB:
		// Get a list of all files in a given directory
		fileVec = readDirectory(runParams.corpusPath);

		// Run buildVocabMap() in parallel on each entry in the file vector
		 tbb::parallel_for(tbb::blocked_range<int>(0, fileVec.size()),
				 VocabBuilder(vmap, fileVec)
		 );

		 vvec = vocabBuilder.pairVocabMap(runParams.vocabSize);
		 vocabBuilder.saveVocabMap(runParams.vocabListFileName, vvec);

		 delete vmap;
		break;
	case BUILD_X:
		// Get a list of all files in a given directory
		fileVec = readDirectory(runParams.corpusPath);
		// Load vocabulary list
		vmap_up = vocabBuilder.readVocabCSV(runParams.vocabListFileName);
		vmap = vmap_up.get();

		// Run buildVocabMap() in parallel on each entry in the file vector
		tbb::parallel_for(tbb::blocked_range<int>(0, fileVec.size()),
				CoMat(xmap, vmap, fileVec)
		);

		cout << "Converting co-occurrance map to sparse matrix with " << xmap->size() << " elements." << endl;
		coMat.xmapTospMat(xmap, vmap_up);
		cout << "Co-occurance matrix has " << coMat.X.n_nonzero << " non-zero elements and "
			 << double(coMat.X.n_nonzero)/double(coMat.X.n_elem)*100 << "% density." << endl;
		coMat.writeX(runParams.XfileName);

		delete xmap;
		delete vmap;
		break;
	case TRAIN:
		cout << "Loading " << runParams.XfileName << endl;
		coMat.readX(runParams.XfileName);

		//cout << "V = " << Xr.getV() << ", Max X: " << max(max(Xr.X)) << ", Min X: " << min(min(Xr.X)) << endl;

		trainer.setup(&coMat, runParams.wordVecDims, runParams.alpha, runParams.xmax, runParams.vecCost);
		trainer.GradDesc(runParams.maxIters, runParams.eta, runParams.momentum, runParams.vecCost);
		if (trainer.SaveWordVecs(runParams.wordVecsFileName)) {
			cerr << "Error saving word vectors!" << endl;
			exit(EXIT_FAILURE);
		} else
			cout << "Wrote word vectors to disk." << endl;
		break;
	case CHECK_GRAD:
		coMat.RandInit(80);
		trainer.setup(&coMat, 15, runParams.alpha, runParams.xmax, runParams.vecCost);
		err = trainer.CheckGrad(runParams.vecCost);
		if (err) {
			cerr << "* NUMERICAL GRADIENT CHECK FAILED!! *" << endl;
		} else
			cout << "Numerical gradient check passed!" << endl;
		break;
	default:
		break;
	}
	cout << "GloVe Execution complete." << endl;

	exit(EXIT_SUCCESS);
}

// Add all files in a directory that are not hidden files or directories themselves
vector<string> readDirectory(string path) {
	vector<string> files;

	try {
		if (boost::filesystem::exists(path)) {
			if (boost::filesystem::is_directory(path)) {
				boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
				for (boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr) {
					if (!boost::filesystem::is_directory(itr->status())) {
						// Extract the file's full path
						string filePath = itr->path().string();
						// Extract just the file's name
						string fileName = itr->path().leaf().string();
						// Make sure file is not hidden (.) and is at least two characters long in name
						if ((fileName.length() > 1) && (fileName[0] != '.')) {
							files.push_back(filePath);
							//cout << fileName << endl;
						}
					}
				}
			}
		}
	} catch (const boost::filesystem::filesystem_error &ex) {
		cout << ex.what() << '\n';
	}

	return files;
}

bool processCommandlineArguments(RunParameters *runParams, RunMode *runMode, int argc, char **argv) {
	/*
	 ******************** Process command line options *************************************
	 */
	try {
		// Define the command line object, and insert a message that describes the program.
		TCLAP::CmdLine cmd(string(GLOVE_DESCRIPTION)+"\n"+LICENSE_TEXT, ' ', to_string(VERSION));

		/*
		 * Optional Parameter Arguments
		 */
		TCLAP::ValueArg<string> corpusPathArg("c", "corpus", "Path to directory containing corpus text files.", false, DEF_CORPUS_PATH, "string", cmd);
		TCLAP::ValueArg<string> vocabListFileNameArg("l", "vlist", "Path to CSV file containing vocabulary list.", false, DEF_VOCAB_LIST_FNAME, "string", cmd);
		TCLAP::ValueArg<string> wordVecsFileNameArg("w", "vec", "Path to file to save word vectors.", false, DEF_WORD_VEC_FNAME, "string", cmd);
		TCLAP::ValueArg<int> vocabSizeArg("s", "vsize", "Vocabulary size (when building up dictionary from corpus).", false, DEF_VOCAB_SIZE, "integer", cmd);
		TCLAP::ValueArg<string> XfileNameArg("", "xfile", "Path to co-occurance matrix (Xij).", false, DEF_X_FNAME, "string", cmd);
		TCLAP::ValueArg<int> wordVecDimsArg("d", "wdim", "Number of dimensions for word vectors.", false, DEF_WORD_VEC_DIMS, "integer", cmd);
		TCLAP::ValueArg<double> alphaArg("a", "alpha", "Factor for weighting function of co-occurance counts.", false, DEF_ALPHA, "double", cmd);
		TCLAP::ValueArg<int> xmaxArg("", "xmax", "Maximum co-occurance count (for weighting function).", false, DEF_XMAX, "integer", cmd);
		TCLAP::ValueArg<double> etaArg("e", "eta", "Learning rate for gradient descent (0 < eta < 1)", false, DEF_ETA, "double", cmd);
		TCLAP::ValueArg<int> maxItersArg("i", "maxiters", "Vocabulary size (when building up dictionary from corpus).", false, DEF_MAX_ITERS, "integer", cmd);
		TCLAP::ValueArg<double> momentumArg("m", "momentum", "Momentum to use during training (0 < m < 1).", false, DEF_MOMENTUM, "double", cmd);
		TCLAP::ValueArg<int> numThreadsArg("n", "nthread", "Number of threads to use when parsing corpus.", false, DEF_NUM_THREADS, "integer", cmd);
		TCLAP::SwitchArg vecCostArg("", "novcost", "Don't use vectorized cost function?", cmd);

		/*
		 * Command arguments (instructing program to do something)
		 */
		vector<TCLAP::Arg*> runModeArgs;
		TCLAP::SwitchArg buildVocabArg("v", "buildVocab", "Build a vocabulary from a corpus.");
		runModeArgs.push_back(&buildVocabArg);

		TCLAP::SwitchArg buildXArg("x", "buildX", "Build the co-occurance matrix from the corpus.");
		runModeArgs.push_back(&buildXArg);

		TCLAP::SwitchArg trainArg("t", "train", "Train word vectors with gradient descent.");
		runModeArgs.push_back(&trainArg);

		TCLAP::SwitchArg checkArg("g", "check", "Check the training algorithm for numerical errors (gradient check).");
		runModeArgs.push_back(&checkArg);
		// Accept only one of the above arguments
		cmd.xorAdd(runModeArgs);

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each arg.
		try {
			runParams->corpusPath = corpusPathArg.getValue();
			runParams->vocabListFileName = vocabListFileNameArg.getValue();
			runParams->wordVecsFileName = wordVecsFileNameArg.getValue();
			runParams->vocabSize = vocabSizeArg.getValue();
			if (rangeCheck(runParams->vocabSize, 9, 100001))
				throw "Vocab size must be between 10 and 100000.";

			runParams->XfileName = XfileNameArg.getValue();
			runParams->wordVecDims = wordVecDimsArg.getValue();
			if (rangeCheck(runParams->wordVecDims, 9, 1001))
				throw "Word vector dimensions must be between 10 and 1000.";

			runParams->alpha = alphaArg.getValue();
			if (rangeCheck(runParams->alpha, 0.0, 1.0))
				throw "Alpha must be between 0 and 1.";

			runParams->xmax = xmaxArg.getValue();
			if (rangeCheck(runParams->xmax, 9, 100001))
				throw "Xmax must be between 10 and 1e5";

			runParams->eta = etaArg.getValue();
			if (rangeCheck(runParams->eta, 0.0, 1.0))
				throw "Eta must be between 0 and 1";

			runParams->maxIters = maxItersArg.getValue();
			if (rangeCheck(runParams->maxIters, 0, 100001))
				throw "Max iterations must be between 1 and 100,000";

			runParams->momentum = momentumArg.getValue();
			if (rangeCheck(runParams->momentum, 0.0, 1.0))
				throw "Momentum must be between 0 and 1";

			runParams->numThreads = numThreadsArg.getValue();
			if (rangeCheck(runParams->numThreads, 0, 129))
				throw "Number of threads must be between 1 and 128";
		} catch (const char * e) {
			cerr << "error: " << e << endl;
			return true;
		}

		runParams->vecCost = !vecCostArg.getValue();

		if (buildVocabArg.getValue())
			*runMode = BUILD_VOCAB;
		else if (buildXArg.getValue())
			*runMode = BUILD_X;
		else if (trainArg.getValue())
			*runMode = TRAIN;
		else if (checkArg.getValue())
			*runMode = CHECK_GRAD;

	} catch (TCLAP::ArgException &e) {
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return true;
	}

	return false;
}

