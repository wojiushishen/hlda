#include "corpus.h"

Corpus::Corpus(const char *dataPath) {
	ifstream data_stream(dataPath, ios::binary);
	if (!data_stream.is_open())
	{
		std::cout << "File " << string(dataPath) << " open failed" << std::endl;
		// exit(0);
	}
	// int D;		//number of Document in the whole tokenlist
	// int W;		//number of Word int the whole tokenlist
	TCount NNZ;	//number of triple in the corpus

	data_stream >> D >> V >> NNZ;
    bool    doc_change_flag = 0;
    TId     pre_doc_id = 1;
    num_tokens = 0;
    Document doc;
    doc.doc_id = 1;
    /*
        generate path information for each document and store it into path
    */
    doc.path = init_path();
    
    doc_offset.push_back(0);
    srand(time(NULL));
    for (int iter=0; iter<NNZ; iter++)
    {   
        TId     doc_id;
        TId     word_id;
        TCount  word_count;
        data_stream >> doc_id >> word_id >> word_count;    
        
        if(doc_id != pre_doc_id)
        {
            doc_offset.push_back(num_tokens);
            corpus.push_back(doc);
            doc.clear();
            doc.path = init_path();
            doc.doc_id = doc_id;    
        }
        num_tokens += word_count;
        // num_tokens = (doc_id != pre_doc_id) ? word_count : (word_count + num_tokens);
        for (int i=0; i< word_count; i++)
        {
            int rand_index = rand() % (L+1);

            doc.word_id.push_back(word_id);
            doc.level_id.push_back(rand_index);
            doc.token_id.push_back(doc.word_id.size());
            if (rand_index == 0) {
                doc.node_id.push_back(0); // topic id 
            }
            else{
                doc.node_id.push_back(doc.path[rand_index-1]); // topic id 
            }
        }
        pre_doc_id = doc_id;
    }

    
    doc_offset.push_back(num_tokens);
    corpus.push_back(doc);
    dump_data<TId>(doc_offset, MACRO_VARIABLE_TO_STRING(doc_offset));

    std::cout << "number of documents " << corpus.size() << std::endl;
    std::cout << "number of tokens " << num_tokens << std::endl;
    print_data(doc.path);
    // printf("number of documents %d\n", corpus.size());
    // printf("number of tokens %d\n", num_tokens);
    // assert(num_tokens != NNZ);
}