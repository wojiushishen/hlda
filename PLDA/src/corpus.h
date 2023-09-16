#ifndef __CORPUS_H
#define __CORPUS_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
// #include <unordered_map>
#include "type.h"
#include "document.h"
#include "const.h"
#include "util.h"
#include <algorithm>

using namespace std;
class Corpus {
public:
    Corpus(const char *dataPath);

    // Corpus(const Corpus &from, int start, int end);

    // std::vector<std::vector<TWord>> w;
    // std::vector<std::string> vocab;
    // std::unordered_map<std::string, TWord> word2id;
    vector<Document>    corpus;
    vector<TCount>      doc_offset;
    TCount D; // number of document
    TCount V; // vocab size
    TCount num_tokens;	//number of triple in the corpus
    // TSize T;
    Document& operator[](int idx)       { return corpus[idx]; }
};



#endif //__CORPUS_H