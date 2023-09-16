#ifndef __TREE_H
#define __TREE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include "corpus.h"
#include "type.h"
#include "util.h"
#include "matrix.cuh"




class Tree{
public:
    /*
        The nodes are stored by level information
    */
    TValue          gama = exp(-6.0);
    TCount          num_nodes;
    TCount          num_extra_nodes;
    TCount          num_total_nodes;
    TCount          max_node_ids;
    TCount          num_documents;
    TId*            h_node_ids;
    TId*            h_node_levels;
    TId*            h_node_ranks;
    TId*            h_child_ids;
    TId*            h_parent_node_ids;

    TCount*         h_extra_node_offset;

    TCount*         h_child_counts;
    TCount*         h_path_count; 

    TId*            h_latent_path;         // size is num_nodes*L
    TCount*         h_latent_path_count; 
    TId*            h_path_pool;
    TValue*         h_latent_path_prob;    // size is num_nodes

    TId*            d_node_ids;    
    TId*            d_node_levels;
    TId*            d_node_ranks;

    TId*            d_parent_node_ids; //size of num_node_ids
    TId*            d_child_ids;

    TId*            d_map_old_to_new;
    TId*            d_map_new_to_old;
    TCount*         d_node_counts;

    TCount*         d_child_counts;
    TCount*         d_path_count;  
    std::vector<TCount>  h_level_offsets;
    TCount*         d_level_offsets;
    TCount*         d_child_offsets;


    //  prob related matrix and vector

    TId*            d_path_pool;         // size is num_nodes*L
    TId*            d_path_pool_ordered;
    TId*            d_path_pool_tmp;


    TId*            d_latent_path;         // size is num_nodes*L
    TCount*         d_latent_path_count;
    TValue*         d_latent_path_prob;    // size is num_nodes

    Tree(Corpus& corpus, cudaStream_t& stream);
    void InitTreeFromCorpus(Corpus& corpus, cudaStream_t& stream);
    void TreeMemoryDeviceToHost(cudaStream_t& stream);
    void TreeMemoryAlloc(cudaStream_t& stream);
    void MapMemoryAlloc(cudaStream_t& stream);
    void TreeMemoryFree(cudaStream_t& stream);
    void MapMemoryFree(cudaStream_t& stream);
    void TreeInitialize(cudaStream_t& stream);
    void CalculateProb(cudaStream_t& stream, int iter);
    void TreeCreate();
    void TreeUpdate();
    void MemFree(cudaStream_t& stream);
};

#endif