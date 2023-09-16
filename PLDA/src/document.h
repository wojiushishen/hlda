#ifndef __DOCUMENT_H
#define __DOCUMENT_H

#include <vector>
#include "type.h"
using namespace std;

// Size a document = 3 ints / token
struct Document {
    TId        doc_id;
    TCount     num_tokens;
    vector<TId>       path;
    vector<TId>       token_id; //size num_tokens
    vector<TId>       word_id;  //size num_tokens
    vector<TId>       level_id; //size num_tokens
    vector<TId>       node_id;  //size num_tokens
    vector<TCount>    doc_level_count;// size L

    void clear();

    // std::vector<int> c;
    // int leaf_id;
    // std::vector<TTopic> z;
    // std::vector<TWord> w;

    // std::vector<TProb> theta;

    // std::vector<TWord> reordered_w;
    // std::vector<int> c_offsets;    // offset for log gamma
    // std::vector<TLen> offsets;

    // bool initialized;

    // void PartitionWByZ(int L, bool compute_c=true);

    // void Check();

    // TLen BeginLevel(int l) { return offsets[l]; }

    // TLen EndLevel(int l) { return offsets[l + 1]; }
};

#endif //__DOCUMENT_H