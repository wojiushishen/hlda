#ifndef __HLDA_H
#define __HLDA_H
#include "corpus.h"
#include "tree.h"
#include "type.h"
#include "const.h"
#include "util.h"
#include <curand.h>
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>

class Para{

public:
	TId*  h_node_visit_count; //changable m_t
	TId*  h_token_doc_id; // d
	TId*  h_token_word_id; // w_dn
	TId*  h_token_level_id;	// z_dn
	TId*  h_token_node_id; // c_d,z_dn
	TId*  h_token_map;
	TId*  h_doc_path;



	TCount*  h_doc_level_count; //a_dl
	TCount*  h_topic_word_count; //b_tv
	TCount*  h_topic_count; //s_t
	TCount*  h_token_counts_in_documents;// N1, N2, ...
    TCount*  h_doc_offsets;

	TValue*   h_beta;
	TValue*   h_phi;
	// TValue*   h_prob_doc_post = new T[]


    TId*  		d_node_visit_count; //changable m_t
	TId*  		d_token_doc_id; // d
	TId*  		d_token_word_id; // w_dn
	TId*  		d_token_level_id;	// z_dn
	TId*  		d_token_node_id; // c_d,z_dn
	TId*        d_token_node_id_tmp;
	TCount* 	d_token_exclusive_scan;
	TCount*		d_token_exclusive_scan_count;

	TId*        d_map_old_to_new; //  switch topic id, need to update d_token_node_id, d_topic_word_count, d_topic_count


	TId*  d_token_word_id_out; // w_dn
	TId*  d_token_level_id_out;	// z_dn

	TId*  d_doc_path;
	TId*  d_token_map;
	TId*  d_token_map_out;


	TCount*  d_doc_level_count; //a_dl
	TCount*  d_topic_word_count; //b_tv
	TCount*  d_topic_count; //s_t
    TCount*  d_doc_offsets;

	curandState* rand_u;

	TId*	 d_path_pool;
	TId*	 d_latent_path;
	TValue*  d_latent_path_prob;


	TCount*   d_token_counts_in_documents;// N1, N2, ...
	TValue*   d_phi;
	// TValue*   d_prob_doc_post = new T[];


    TCount n_words;
    TCount n_documents;
    TCount n_topics;
    int n_tokens;
    TCount n_levels;
	
	// hyperPara alpha, beta,
	
	// TValue beta0 = exp(-4.0);
	// TValue beta[L+1] = {beta0, 0.5*beta0, 0.25*beta0, 0.25*beta0};
	TValue beta[L+1] = {1, 0.5, 0.3, 0.07};
	TValue* d_beta;
	// beta={beta0, 0.5*beta0, 0.25*beta0, 0.25*beta0};
	// beta[0] = beta0;
	// beta[1] = 0.5*beta0;
	// beta[2] = 0.25*beta0;
	// beta[3] = 0.25*beta0;
	TValue alpha[L+1] = {0.3, 0.3, 0.3, 0.3};
	TValue* d_alpha;

	TValue*  d_perp;


    Para(Corpus& corpus, Tree& tree, cudaStream_t& stream);
	void Update_iteration(Tree& tree, cudaStream_t& stream);
	void MemoryHostToDevice(cudaStream_t& stream);
    void MemoryDeviceToHost(cudaStream_t& stream);
    void MemoryAlloc(cudaStream_t& stream);
    void MemoryFree(cudaStream_t& stream);
	void SampleZ(cudaStream_t& stream);
	void Sample_process(cudaStream_t& stream, int iter);

};

#endif