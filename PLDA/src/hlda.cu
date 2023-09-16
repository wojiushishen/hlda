#include "hlda.h"
#include "util.h"

Para::Para(Corpus& corpus, Tree& tree, cudaStream_t& stream){

    n_words = corpus.V;
    n_documents = corpus.D;
    n_topics = tree.num_nodes + 1;
    n_tokens = corpus.num_tokens;
    n_levels = L+1;
    h_doc_offsets = &corpus.doc_offset[0];

    h_token_doc_id 	    = new TId[n_tokens]; // d
	h_token_word_id     = new TId[n_tokens]; // w_dn
	h_token_level_id    = new TId[n_tokens];	// z_dn
	h_token_node_id     = new TId[n_tokens]; // c_d,z_dn
    // h_token_map         = new TId[n_tokens];
    
    // h_doc_path          = new TId[n_levels*n_documents];
    h_doc_level_count   = new TCount[n_levels*n_documents];
    h_topic_count       = new TCount[n_topics];
    h_topic_word_count  = new TCount[n_topics*n_words];

    d_path_pool         = &tree.d_path_pool[0];
	d_latent_path       = &tree.d_latent_path[0];
	d_latent_path_prob  = &tree.d_latent_path_prob[0];
    d_map_old_to_new    = &tree.d_map_old_to_new[0];

    memset(h_topic_count, 0, sizeof(TCount)*n_topics);
    memset(h_doc_level_count, 0, sizeof(TCount)*n_levels*n_documents);
    memset(h_topic_word_count, 0, sizeof(TCount)*n_topics*n_words);

    int count =0;
    for(int i=0; i<corpus.corpus.size(); i++){
        for (int j=0; j < corpus.corpus[i].token_id.size(); j++){
            h_token_doc_id[j + count] = corpus.corpus[i].doc_id-1;
            h_token_word_id[j + count] = corpus.corpus[i].word_id[j]-1;
            h_token_level_id[j + count] = corpus.corpus[i].level_id[j];
            h_token_node_id[j + count] = corpus.corpus[i].node_id[j];
            h_doc_level_count[h_token_doc_id[j + count]*n_levels + h_token_level_id[j + count]] += 1;
            h_topic_count[corpus.corpus[i].node_id[j]] += 1;
            h_topic_word_count[corpus.corpus[i].node_id[j]*n_words + h_token_word_id[j + count]] += 1;
        }

        // h_doc_path[i*n_levels] = 0;
        // for (int j=1; j<n_levels; j++){
        //     h_doc_path[i*n_levels+j] = corpus.corpus[i].path[j-1];
        // }
        count += corpus.corpus[i].token_id.size();
    }

    // for (int i=0; i<n_tokens; i++){
    //     h_token_map[i]=i;
    // }
    // dump_data_ptr<TId>(h_token_doc_id, n_tokens, MACRO_VARIABLE_TO_STRING(h_token_doc_id));
    // dump_data_ptr<TId>(h_token_word_id, n_tokens, MACRO_VARIABLE_TO_STRING(h_token_word_id));
    // dump_data_ptr<TId>(h_token_level_id, n_tokens, MACRO_VARIABLE_TO_STRING(h_token_level_id));
    // dump_data_ptr<TId>(h_token_node_id, n_tokens, MACRO_VARIABLE_TO_STRING(h_token_node_id)); 
    // dump_mat_ptr<TCount>(h_doc_level_count, n_documents, n_levels, MACRO_VARIABLE_TO_STRING(h_doc_level_count));
    // dump_mat_ptr<TCount>(h_topic_word_count, n_topics, n_words, MACRO_VARIABLE_TO_STRING(h_topic_word_count));
    // dump_data_ptr<TCount>(h_topic_count, n_topics, MACRO_VARIABLE_TO_STRING(h_topic_count));
    // dump_mat_ptr<TId>(h_doc_path, n_documents, n_levels, MACRO_VARIABLE_TO_STRING(h_doc_path));
}



void Para::MemoryAlloc(cudaStream_t& stream){
    //---------------------------------Calculate perplexity------------------------
    gpuErrchk(cudaMallocAsync(&d_perp,            sizeof(TValue)*n_tokens, stream));
    // gpuErrchk(cudaMallocAsync(&d_perp,            sizeof(TValue)*n_tokens, stream));
    //---------------------------------Calculate perplexity------------------------

    gpuErrchk(cudaMallocAsync(&d_token_doc_id,    sizeof(TId)*n_tokens, stream));
    gpuErrchk(cudaMallocAsync(&d_token_word_id,   sizeof(TId)*n_tokens, stream));
    gpuErrchk(cudaMallocAsync(&d_token_level_id,  sizeof(TId)*n_tokens, stream));
    gpuErrchk(cudaMallocAsync(&d_token_node_id,   sizeof(TId)*n_tokens, stream));
    gpuErrchk(cudaMallocAsync(&d_token_node_id_tmp, sizeof(TId)*n_tokens, stream));
    gpuErrchk(cudaMallocAsync(&d_token_exclusive_scan,   sizeof(TCount)*n_tokens, stream));
    gpuErrchk(cudaMallocAsync(&d_token_exclusive_scan_count,   sizeof(TCount)*n_tokens, stream));

    // gpuErrchk(cudaMallocAsync(&d_token_word_id_out,   sizeof(TId)*n_tokens, stream));
    // gpuErrchk(cudaMallocAsync(&d_token_level_id_out,  sizeof(TId)*n_tokens, stream));

    gpuErrchk(cudaMallocAsync(&d_doc_level_count,     sizeof(TCount)*n_levels*n_documents, stream));
    gpuErrchk(cudaMallocAsync(&d_topic_word_count,    sizeof(TCount)*n_topics*n_words, stream));
    // gpuErrchk(cudaMallocAsync(&d_doc_path,            sizeof(TId)*n_documents*n_levels, stream));

    gpuErrchk(cudaMallocAsync(&d_topic_count,         sizeof(TCount)*n_topics, stream));
    gpuErrchk(cudaMallocAsync(&d_doc_offsets,         sizeof(TCount)*(n_documents + 1), stream));
    gpuErrchk(cudaMallocAsync(&d_beta, sizeof(TValue)*(n_levels), stream));
    gpuErrchk(cudaMallocAsync(&d_alpha,sizeof(TValue)*(n_levels), stream));    
    gpuErrchk(cudaMalloc(&rand_u, sizeof(curandState)*GridDim*BlockDim));

}

void Para::MemoryHostToDevice(cudaStream_t& stream){
      gpuErrchk(cudaMemcpyAsync(d_token_doc_id, h_token_doc_id, sizeof(TId)*n_tokens, cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_token_word_id, h_token_word_id, sizeof(TId)*n_tokens, cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_token_level_id, h_token_level_id, sizeof(TId)*n_tokens, cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_token_node_id, h_token_node_id, sizeof(TId)*n_tokens, cudaMemcpyHostToDevice, stream));

      gpuErrchk(cudaMemcpyAsync(d_doc_level_count, h_doc_level_count, sizeof(TCount)*n_levels*n_documents, cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_topic_word_count, h_topic_word_count, sizeof(TCount)*n_topics*n_words, cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_topic_count, h_topic_count, sizeof(TCount)*n_topics, cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_doc_offsets, h_doc_offsets, sizeof(TCount)*(n_documents + 1), cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_alpha, alpha, sizeof(TValue)*(n_levels), cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(d_beta,  beta,  sizeof(TValue)*(n_levels), cudaMemcpyHostToDevice, stream));
    //   gpuErrchk(cudaMemcpyAsync(d_doc_path, h_doc_path, sizeof(TId)*n_documents*n_levels, cudaMemcpyHostToDevice, stream));
}


void Para::MemoryDeviceToHost(cudaStream_t& stream){
      gpuErrchk(cudaMemcpyAsync(h_token_doc_id, d_token_doc_id, sizeof(TId)*n_tokens, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_token_word_id, d_token_word_id, sizeof(TId)*n_tokens, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_token_level_id, d_token_level_id, sizeof(TId)*n_tokens, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_token_node_id, d_token_node_id, sizeof(TId)*n_tokens, cudaMemcpyDeviceToHost, stream));

      gpuErrchk(cudaMemcpyAsync(h_doc_level_count, d_doc_level_count, sizeof(TCount)*n_levels*n_documents, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_topic_word_count, d_topic_word_count, sizeof(TCount)*n_topics*n_words, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_topic_count, d_topic_count,  sizeof(TCount)*n_topics, cudaMemcpyDeviceToHost, stream));
}

__global__ void perp_kernel(TValue* perp,
                            TCount* doc_level_count,
                            TCount* topic_word_count,
                            TCount* topic_count,
                            TValue* alpha,
                            TValue* beta,
                            TId*    token_doc_id,
                            TId*    token_word_id,
                            TId*    token_node_id,
                            TId*    token_level_id,
                            TCount* doc_offsets,
                            TCount  n_words,
                            TCount  n_levels,
                            TCount  n_tokens)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //
    for (TId token_id = tid; token_id < n_tokens; token_id += blockDim.x * gridDim.x){
        TId doc_id      = token_doc_id[token_id];
        TId word_id     = token_word_id[token_id];
        TId level_id    = token_level_id[token_id];
        TId node_id     = token_node_id[token_id];
        TCount n_d      = doc_offsets[doc_id+1] - doc_offsets[doc_id];
        TValue local_doc_level_count = doc_level_count[doc_id*n_levels+level_id]+ alpha[level_id];
        TValue local_topic_word_count = topic_word_count[node_id*n_words+ word_id]+ beta[level_id];
        TValue local_topic_count = topic_count[node_id] + n_words * beta[level_id];
        perp[token_id]  = log(local_doc_level_count * local_topic_word_count/local_topic_count/(n_d + n_levels*alpha[level_id]));
    }
}

TValue perp_sum(TValue* perp,
                TCount  n_tokens,
                cudaStream_t&  stream)
{
    auto sync_exec_policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<TValue> perp_ptr(perp);
    TValue result = thrust::reduce(sync_exec_policy, perp_ptr, perp_ptr + n_tokens, 0);
    cudaStreamSynchronize(stream);
    result = result/n_tokens;
    return result;
}




__global__ void SampleZ_kernel( TCount* doc_level_count,
                                TValue* alpha,
                                TCount* topic_word_count,
                                TValue* beta,
                                TCount* topic_count,
                                TCount  n_words,
                                TCount  n_levels,
                                TCount  n_tokens,
                                TCount  n_documents,
                                TId*    token_doc_id,
                                TId*    token_word_id,
                                TId*    token_node_id,
                                TId*    token_level_id,
                                TId*    path_pool,
                                curandState*  rand_u)
{
    // each warp proceeds a document, each L threads proceed a token 
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //
    __shared__ TValue p[(L+1)* BlockDim];

    
    for (TId token_id = tid; token_id < n_tokens; token_id += blockDim.x * gridDim.x){
        TId doc_id      = token_doc_id[token_id];
        TId word_id     = token_word_id[token_id];
        TId level_id    = token_level_id[token_id];
        TId node_id     = token_node_id[token_id];
        TValue u        = curand_uniform(&(rand_u[tid])) / 1.00001;

        //bugs here, need to calculate sum first
        // p_sum[threadIdx.x] = 0.0;
        for (int i= 0; i<n_levels; i++){
            int tmp = (i==level_id);
            // int topic = (i < 1)? 0 : doc_path[doc_id*n_levels + i];
            int topic = node_id;
            TValue tmp1 = doc_level_count[doc_id*n_levels+i]-tmp + alpha[i];
            TValue tmp2 = topic_word_count[topic*n_words+ word_id]-tmp + beta[i];
            TValue tmp3 = topic_count[topic] - tmp + n_words * beta[i];
            if(topic_word_count[topic*n_words+ word_id]-tmp < 0) printf("topic_word_count, tmp, doc_id, word_id, level_id, node_id: %d, %d, %d, %d, %d, %d\n",topic_word_count[topic*n_words+ word_id], tmp, doc_id, word_id, level_id, node_id);
            if(topic_count[topic] - tmp < 0) printf("topic_count, tmp, doc_id, word_id, level_id: %d, %d, %d, %d, %d\n",topic_count[topic], tmp, doc_id, word_id, level_id);   
            if(doc_level_count[doc_id*n_levels+i]-tmp< 0) printf("doc_level_count, tmp, doc_id, word_id, level_id: %d, %d, %d, %d, %d\n",doc_level_count[doc_id*n_levels+i], tmp, doc_id, word_id, level_id);   
            TValue p_pre = (i < 1)? 0.0 : p[threadIdx.x* n_levels + (i - 1)];
            p[threadIdx.x* n_levels + i] = p_pre + tmp1*tmp2/tmp3;
        }
        u = p[threadIdx.x* n_levels + n_levels - 1] * u;
        for (int i= 0; i< n_levels; i++)
        {   
            TValue lower_bound = 0.0;
            if (i > 0) lower_bound = p[threadIdx.x* n_levels + i - 1];
            TValue upper_bound = p[threadIdx.x* n_levels + i];
            if(u > lower_bound && u < upper_bound) {
                token_level_id[token_id] = i;
                token_node_id[token_id] = (i < 1)? 0 :path_pool[doc_id*n_levels + i];
                break;
            }
        }
        
    }
}


__global__ void Update_D_W_kernel(      TCount*         doc_level_count,
                                        TCount*         topic_word_count,
                                        TCount          n_words,
                                        TCount          n_levels,
                                        TCount          n_tokens,
                                        TCount          n_topics,
                                        TId*            token_doc_id,
                                        TId*            token_word_id,
                                        TId*            token_node_id,
                                        TId*            token_level_id)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //
    for (TId token_id = tid; token_id < n_tokens; token_id += blockDim.x * gridDim.x){
        TId doc_id      = token_doc_id[token_id];
        TId word_id     = token_word_id[token_id];
        TId level_id    = token_level_id[token_id];
        TId node_id     = token_node_id[token_id];
        atomicAdd(&doc_level_count[doc_id * n_levels + level_id], 1);
        // if (node_id * n_words + word_id < 0 || node_id * n_words + word_id  > n_topics* n_words){ printf("node_id, word_id: %d, %d, %d, %d\n", node_id, word_id, n_words, n_topics); };
        atomicAdd(&topic_word_count[node_id * n_words + word_id], 1);
    }
    __syncthreads();
}

__global__ void Update_S_kernel(        TCount*         topic_word_count,
                                        TCount*         topic_count,
                                        TCount          n_words,
                                        TCount          n_topics)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //
    for (TId topic_id = tid; topic_id < n_topics; topic_id += blockDim.x * gridDim.x){
        for (TId word_id = 0; word_id < n_words; word_id++){
            topic_count[topic_id] += topic_word_count[topic_id * n_words + word_id];
        }
    }
}



typedef thrust::tuple<TId, TId, TId, TCount> IntTuple;
typedef thrust::tuple<TId, TId, TId> ScanTuple;
struct sort_functor
{
      __host__ __device__
      bool operator()(const IntTuple& t1, const IntTuple& t2){
            if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
            if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;
            if (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
            if (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;
            if (thrust::get<2>(t1) < thrust::get<2>(t2)) return true;
            if (thrust::get<2>(t1) > thrust::get<2>(t2)) return false;
            if (thrust::get<3>(t1) > thrust::get<3>(t2)) return true;
            if (thrust::get<3>(t1) < thrust::get<3>(t2)) return false;
            return false;
      }
};


// __global__ void UpdateTopicWordCount_kernel(TCount* topic_word_count,
//                                             TCount* topic_word_count_new,
//                                             TCount* topic_count,
//                                             TCount* topic_count_new,
//                                             TId*    map_old_to_new,
//                                             int     n_topics,
//                                             int     n_words)
// {
//     int tid = blockIdx.x*blockDim.x + threadIdx.x;
//     for (int i = tid; i < n_topics * n_words; i+= blockDim.x * gridDim.x ){
//         int old_topic_id    = i / n_words;
//         int word_id         = i % n_words;
//         int new_topic_id    = map_old_to_new[old_topic_id];
//         int new_position    = new_topic_id * n_words + word_id;
//         topic_word_count_new[new_position] = topic_word_count[i];
//         topic_count_new[new_topic_id] = topic_count[old_topic_id];
//     }
// }

__global__ void Update_W_kernel(TId*    token_word_id, // wd
                                TId*    token_node_id,
                                TCount* topic_word_count, // b_tv
                                int     n_tokens,
                                int     n_topics,
                                int     n_words)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = tid; i < n_tokens; i+= blockDim.x * gridDim.x ){
        int new_position    = token_node_id[i] * n_words + token_word_id[i];
        atomicAdd(&topic_word_count[new_position], 1);
    }
}




// __global__ void Update_kernel(TCount* topic_word_count,
//                               TCount* topic_word_count_new,
//                               TCount* topic_count,
//                               TCount* topic_count_new,
//                               TId*    map_old_to_new,
//                               int     n_topics,
//                               int     n_words)
// {
//     int tid = blockIdx.x*blockDim.x + threadIdx.x;
//     for (int i = tid; i < n_topics * n_words; i+= blockDim.x * gridDim.x ){
//         int old_topic_id    = i / n_words;
//         int word_id         = i % n_words;
//         int new_topic_id    = map_old_to_new[old_topic_id];
//         int new_position    = new_topic_id * n_words + word_id;
//         topic_word_count_new[new_position] = topic_word_count[i];
//         topic_count_new[new_topic_id] = topic_count[old_topic_id];
//     }
// }



void Update_C_M(    TId*           token_doc_id,
                    TId*           token_word_id, // wd
                    TId*           token_node_id, 
                    TId*           token_level_id, // zd
                    TCount*        token_exclusive_scan,
                    TCount*        token_exclusive_scan_count,
                    TCount*        topic_word_count,
                    TCount*        topic_count,
                    cudaStream_t&  stream,
                    TCount         n_tokens,
                    TCount         n_topics,
                    TCount         n_words)
{
    auto sync_exec_policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<TId> token_doc_id_ptr(token_doc_id);
    thrust::device_ptr<TId> token_word_id_ptr(token_word_id);
    thrust::device_ptr<TId> token_level_id_ptr(token_level_id);
    thrust::device_ptr<TId> token_node_id_ptr(token_node_id);
    thrust::device_ptr<TCount> token_exclusive_scan_ptr(token_exclusive_scan);
    thrust::device_ptr<TCount> token_exclusive_scan_count_ptr(token_exclusive_scan_count);
    thrust::constant_iterator<int> const_one(1);

    {
        auto key_begin = thrust::make_zip_iterator(thrust::make_tuple(token_doc_id_ptr, token_level_id_ptr, token_word_id_ptr));
        auto key_end = thrust::make_zip_iterator(thrust::make_tuple(token_doc_id_ptr + n_tokens, token_level_id_ptr + n_tokens, token_word_id_ptr + n_tokens));
        thrust::sort_by_key(sync_exec_policy, key_begin, key_end, token_node_id_ptr);
        thrust::exclusive_scan_by_key(sync_exec_policy, key_begin, key_end, const_one, token_exclusive_scan_ptr);
    }

    {
        auto key_begin = thrust::make_zip_iterator(thrust::make_tuple(token_doc_id_ptr, token_level_id_ptr, token_word_id_ptr, token_exclusive_scan_ptr));
        auto key_end = thrust::make_zip_iterator(thrust::make_tuple(token_doc_id_ptr + n_tokens, token_level_id_ptr + n_tokens, token_word_id_ptr + n_tokens, token_exclusive_scan_ptr + n_tokens));
        thrust::sort(sync_exec_policy, key_begin, key_end, sort_functor());
        // thrust::inclusive_scan_by_key(sync_exec_policy, key_begin, key_end, token_exclusive_scan_ptr, token_exclusive_scan_ptr);
    }

    {
        thrust::equal_to<ScanTuple>    binary_pred;
        thrust::maximum<TCount>        binary_op;
        auto key_begin = thrust::make_zip_iterator(thrust::make_tuple(token_doc_id_ptr, token_level_id_ptr, token_word_id_ptr));
        auto key_end = thrust::make_zip_iterator(thrust::make_tuple(token_doc_id_ptr + n_tokens, token_level_id_ptr + n_tokens, token_word_id_ptr + n_tokens));
        thrust::inclusive_scan_by_key(sync_exec_policy, key_begin, key_end, token_exclusive_scan_ptr, token_exclusive_scan_count_ptr, binary_pred, binary_op);
    }

}


void MapNewTreeId(      TId*           token_node_id, 
                        TId*           token_node_id_tmp, 
                        TId*           map_old_to_new,
                        cudaStream_t&  stream,
                        TCount         n_tokens)
{
    auto sync_exec_policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<TId>     token_node_id_ptr(token_node_id);
    thrust::device_ptr<TId>     token_node_id_tmp_ptr(token_node_id_tmp);
    thrust::device_ptr<TId>     map_old_to_new_ptr(map_old_to_new);
    thrust::gather(sync_exec_policy, token_node_id_ptr, token_node_id_ptr + n_tokens, map_old_to_new_ptr, token_node_id_tmp_ptr);
    gpuErrchk(cudaMemcpyAsync(token_node_id, token_node_id_tmp, sizeof(TId)*n_tokens, cudaMemcpyDeviceToDevice, stream));
}



__global__ void initRandState(curandState *state)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(clock() + tid, tid, 0, &state[tid]);
}

__global__ void SampleC_kernel( TId*           token_doc_id,
                                TId*           token_word_id, // wd
                                TId*           token_node_id,
                                TId*           token_level_id, // zd
                                TCount*        token_exclusive_scan,
                                TCount*        token_exclusive_scan_count,
                                TId*           path_pool,
                                TId*           latent_path,
                                TValue*        latent_path_prob,
                                TCount*        topic_word_count, // b_tv
                                TValue*        beta,
                                TCount*        topic_count,
                                TCount*        doc_offsets,
                                TCount         n_topics,
                                TCount         n_words,
                                TCount         n_levels,
                                TCount         n_documents,
                                curandState*   rand_u                            
                                )
{
    // each block proceeds a document, each L threads proceed a token 
    // int tid = threadIdx.x + blockIdx.x * blockDim.x; //
    extern __shared__ TValue p[];

    // int laneId = threadIdx.x % 32;
	// int localId = threadIdx.x / 32;
    TValue* p_array         = (TValue*) &p;// size n_topics
    // TValue* p_array_block_sum   = (TValue*) &p_array[n_topics];
    int size_p_array        = ((n_topics-1)/blockDim.x + 1) * blockDim.x;
    TValue* p_array_copy    = (TValue*) &p_array[size_p_array]; 
    TValue* tmp_sum         = (TValue*) &p_array_copy[blockDim.x];
    TValue* rand            = (TValue*) &tmp_sum[1];
    TId*    chosen_path     = (TId*)    &rand[1];
    // TValue* p_array_copy    = (TValue*) &p_array[n_topics]; 
    // TValue* p_array_warp_sum    = (TValue*) &p_array[n_topics]; // size 32 * blockDim.x / 32
    // TValue* p_path  = (TValue*) &p_sum[n_topics];
    // // TId* path_topic  = (TId*) &p_path[n_topics];
    // for (TId i = threadIdx.x; i < n_topics; i += blockDim.x){
    //     p_path[i] = latent_path_prob[i];
    // }
    // __syncthreads();
    for (TId i = blockIdx.x; i < n_documents; i += gridDim.x){
    // calculate log(fc)
        for (TId j = threadIdx.x; j < ((n_topics-1)/blockDim.x + 1) * blockDim.x; j += blockDim.x)
        {
            p_array[j] = 0.0;
        }
        __syncthreads();
        int doc_start = doc_offsets[i];
        int doc_end = doc_offsets[i+1];
        for (TId j = threadIdx.x; j < n_topics; j += blockDim.x){
            TValue p_fc = 0.0;
            TValue ht = 0.0;
            int total_exclusive_scan_count = 0;
            int old_level_id = token_level_id[doc_start];
            int old_node_id = token_node_id[doc_start];
            int old_topic = (old_level_id < 1)? 0 : latent_path[(old_level_id-1)*n_topics + j];

            for (TId k = doc_start; k < doc_end; k++){
                int word_id = token_word_id[k];
                int node_id = token_node_id[k];
                int level_id = token_level_id[k];
                if(old_topic < 0) printf("word_id, node_id, level_id, old_level_id, n_topics, k-doc_start, doc_start, latent_path: %d, %d, %d, %d, %d, %d, %d, %d\n", word_id, node_id, level_id, old_level_id, n_topics, k-doc_start, doc_start, (old_level_id-1)*n_topics + j);
                int tmp_topic_count= (old_topic > (n_topics - 1))? 0 : topic_count[old_topic];
                tmp_topic_count= (old_topic == old_node_id)? (tmp_topic_count - total_exclusive_scan_count) : tmp_topic_count;

                int topic = (level_id < 1)? 0 : latent_path[(level_id-1)*n_topics + j];
                if (level_id != old_level_id){
                    // if (old_node_id != old_topic) total_exclusive_scan_count = 0;
                    ht += lgamma(tmp_topic_count + n_words * beta[old_level_id]) - lgamma(tmp_topic_count + total_exclusive_scan_count+ n_words * beta[old_level_id]);
                    if(tmp_topic_count < 0) printf("tmp_topic_count, total_exclusive_scan_count: %d, %d\n",tmp_topic_count, total_exclusive_scan_count);
                    total_exclusive_scan_count = 0; 
                }
                total_exclusive_scan_count += 1;                
                int exclusive_scan = token_exclusive_scan[k];
                int exclusive_scan_count = (node_id == topic)? (token_exclusive_scan_count[k] + 1) : 0;
                // total_exclusive_scan_count += exclusive_scan_count;
                
                if(topic < 0) printf("leve_id, n_topics, j: %d, %d, %d\n", level_id, n_topics, j); 
                int tmp_topic_word_count = (topic > (n_topics - 1))? 0 : topic_word_count[topic*n_words+ word_id];
                TValue topic_word_count_without_d = tmp_topic_word_count - exclusive_scan_count + exclusive_scan + beta[level_id];
                // if (i == 0 && threadIdx.x == 7) printf("tmp_topic_word_count, exclusive_scan_count, exclusive_scan, topic_word_count_without_d, node_id, topic: %d, %d, %d, %lf, %d, %d\n", tmp_topic_word_count, exclusive_scan_count, exclusive_scan, topic_word_count_without_d, node_id, topic); 
                if(tmp_topic_word_count - exclusive_scan_count < 0) printf("tmp_topic_word_count, exclusive_scan_count: %d, %d\n", tmp_topic_word_count, exclusive_scan_count);
                    
                p_fc += log(topic_word_count_without_d);
                old_level_id = level_id;
                old_topic = topic;
                old_node_id =node_id;

                tmp_topic_count = (topic > (n_topics - 1))? 0 : topic_count[topic];
                tmp_topic_count= (topic == node_id)? (tmp_topic_count - total_exclusive_scan_count) : tmp_topic_count;

                if(k == (doc_end-1)){
                    // if (node_id != topic) total_exclusive_scan_count = 0;
                    ht += lgamma(tmp_topic_count + n_words * beta[level_id]) - lgamma(tmp_topic_count + total_exclusive_scan_count + n_words * beta[level_id]);
                    if(tmp_topic_count < 0) printf("tmp_topic_count2: %d\n",tmp_topic_count);                    
                }
            }
            // p_fc += log(gama())
            // p_array[j] = latent_path_prob[j] * exp(p_fc + ht);
            // p_array[j] = latent_path_prob[j] * exp(p_fc + ht);
            p_array[j] = p_fc + ht;
            // p_array[j] = ht;
            // printf("Value p_array[%d]=%lf, %lf\n", j,  p_fc, ht);
        }
        __syncthreads();

        for (TId j = threadIdx.x; j < n_topics; j += blockDim.x){
            p_array[j] = log(latent_path_prob[j]) + p_array[j];
        }

        // if(threadIdx.x == 0) {
        //     tmp_sum[0] = 0.0;
        //     chosen_path[0] = 1;
        // }
        __syncthreads();
        if(threadIdx.x == 0){
            TValue max_value = -9999999999.0;
            TValue rand_number = curand_uniform(&(rand_u[blockIdx.x]));
            TValue tmp_pre_sum = 0.0;
            for(TId j = 0; j < n_topics; j++){
                max_value = (max_value < p_array[j]) ? p_array[j] : max_value;
            }
            // tmp_sum[0] = max_value;
            for(TId j = 0; j < n_topics; j++){
                if (p_array[j] - max_value > -20) {
                    p_array[j] = exp(p_array[j] - max_value);
                }
                else{
                    p_array[j] = 0.0;
                }
            }
            for(TId j = 0; j < n_topics; j++)
            {
                p_array[j] += tmp_pre_sum;
                tmp_pre_sum = p_array[j];
            }
            tmp_pre_sum = tmp_pre_sum * curand_uniform(&(rand_u[blockIdx.x])) / 1.000001;

            // TValue lower_bound = 0.0;
            // TValue upper_bound;
            for(TId j = 0; j < n_topics; j++)
            {
                if (p_array[j] > tmp_pre_sum) {
                    chosen_path[0] = j;
                    break;
                }

                // upper_bound = p_array[j];
                // if (tmp_pre_sum < upper_bound && tmp_pre_sum > lower_bound){

                //     break;
                // }
                // lower_bound = upper_bound;
            }

            for(int j = 0; j < (n_levels - 1); j++){
                path_pool[j * n_documents + i] = latent_path[ j * n_topics + chosen_path[0]];
                if (path_pool[j * n_documents + i] < 1 ) printf("Wrong!\n");
            }

        }
        __syncthreads();

        // Update token list
        for (TId j = doc_start + threadIdx.x; j < doc_end; j += blockDim.x)
        {
            if (token_level_id[j] > 0) token_node_id[j] = path_pool[(token_level_id[j] - 1)*n_documents + i];
        }
        __syncthreads();

        // __syncthreads();
        // for (TId j = threadIdx.x; j < ((n_topics-1)/blockDim.x + 1) * blockDim.x; j += blockDim.x){
        //     // TValue tmp_sum = 0.0;
        //     for (TId k = 1 ; k < blockDim.x; k *= 2){
        //         TValue tmp_p = p_array[j];
        //         p_array_copy[threadIdx.x] = 0.0;
        //         __syncthreads();
        //         if(threadIdx.x + k < blockDim.x) p_array_copy[threadIdx.x + k] = tmp_p;
        //         __syncthreads();
        //         p_array[j] += p_array_copy[threadIdx.x];
        //         __syncthreads();

        //     }
        //     p_array[j] += tmp_sum[0];
        //     __syncthreads();
        //     if(threadIdx.x == (blockDim.x - 1)) tmp_sum[0] = p_array[j];
        //     __syncthreads();
        // }
        // // if (i == 0 && threadIdx.x == 0){
        // //     for(int k=0; k < n_topics ;k++){
        // //         // printf("Value p_array[%d]=%lf\n", k, p_array[k]/p_array[n_topics - 1]);
        // //         printf("Value p_array[%d]=%lf\n", k, p_array[k]);
        // //     }
        // // }
        // __syncthreads();
        // if (threadIdx.x == 0) rand[0] = p_array[n_topics - 1] * curand_uniform(&(rand_u[blockIdx.x])) / 1.00001;
        // __syncthreads();
        // for (TId j = threadIdx.x; j < size_p_array; j += blockDim.x){
        //     TValue lower_bound = 0.0;
        //     if (j > 0) lower_bound = p_array[j -1];
        //     TValue upper_bound = p_array[j];
        //     if (j < n_topics && rand[0] > lower_bound && rand[0] < upper_bound ) chosen_path[0] = j;
        //     __syncthreads();
        // }
        // __syncthreads();
        // if(threadIdx.x == 0){
        //     for(int j = 0; j < (n_levels - 1); j++){
        //         path_pool[j * n_documents + i] = latent_path[ j * n_topics + chosen_path[0]];
        //         if (path_pool[j * n_documents + i] < 1 ) printf("Wrong!\n");
        //     }
        // }
        // __syncthreads();
        
        // // Update token list
        // for (TId j = doc_start + threadIdx.x; j < doc_end; j += blockDim.x)
        // {
        //     if (token_level_id[j] > 0) token_node_id[j] = path_pool[(token_level_id[j] - 1)*n_documents + i];
        // }
        // __syncthreads();


/*
        //Sampling from p_array
        if(threadIdx.x == 0) {
            tmp_sum[0] = 0.0;
            chosen_path[0] = 1;
        }

        __syncthreads();
        for (TId j = threadIdx.x; j < ((n_topics-1)/blockDim.x + 1) * blockDim.x; j += blockDim.x){
            // TValue tmp_sum = 0.0;
            for (TId k = 1 ; k < blockDim.x; k *= 2){
                TValue tmp_p = p_array[j];
                p_array_copy[threadIdx.x] = 0.0;
                __syncthreads();
                if(threadIdx.x + k < blockDim.x) p_array_copy[threadIdx.x + k] = tmp_p;
                __syncthreads();
                p_array[j] += p_array_copy[threadIdx.x];
                __syncthreads();

            }
            p_array[j] += tmp_sum[0];
            __syncthreads();
            if(threadIdx.x == (blockDim.x - 1)) tmp_sum[0] = p_array[j];
            __syncthreads();
        }
        // if (i == 0 && threadIdx.x == 0){
        //     for(int k=0; k < n_topics ;k++){
        //         // printf("Value p_array[%d]=%lf\n", k, p_array[k]/p_array[n_topics - 1]);
        //         printf("Value p_array[%d]=%lf\n", k, p_array[k]);
        //     }
        // }
        __syncthreads();
        if (threadIdx.x == 0) rand[0] = p_array[n_topics - 1] * curand_uniform(&(rand_u[blockIdx.x])) / 1.00001;
        __syncthreads();
        for (TId j = threadIdx.x; j < size_p_array; j += blockDim.x){
            TValue lower_bound = 0.0;
            if (j > 0) lower_bound = p_array[j -1];
            TValue upper_bound = p_array[j];
            if (j < n_topics && rand[0] > lower_bound && rand[0] < upper_bound ) chosen_path[0] = j;
            __syncthreads();
        }
        __syncthreads();
        if(threadIdx.x == 0){
            for(int j = 0; j < (n_levels - 1); j++){
                path_pool[j * n_documents + i] = latent_path[ j * n_topics + chosen_path[0]];
                if (path_pool[j * n_documents + i] < 1 ) printf("Wrong!\n");
            }
        }
        __syncthreads();
        
        // Update token list
        for (TId j = doc_start + threadIdx.x; j < doc_end; j += blockDim.x)
        {
            if (token_level_id[j] > 0) token_node_id[j] = path_pool[(token_level_id[j] - 1)*n_documents + i];
        }
        __syncthreads();
        // TValue p_prefix_sum = 0.0;
        // for (TId j = threadIdx.x; j < ((n_topics-1)/32 + 1) * 32; j += blockDim.x){
        //     p_prefix_sum = 0.0;
        //     p_prefix_sum = p_final[j] * (j < n_topics);
        //     p_prefix_sum += __shfl_up_sync(p_prefix_sum, 1, 32)*(laneId >= 1);
		// 	p_prefix_sum += __shfl_up_sync(p_prefix_sum, 2, 32)*(laneId >= 2);
		// 	p_prefix_sum += __shfl_up_sync(p_prefix_sum, 4, 32)*(laneId >= 4);
		// 	p_prefix_sum += __shfl_up_sync(p_prefix_sum, 8, 32)*(laneId >= 8);
		// 	p_prefix_sum += __shfl_up_sync(p_prefix_sum, 16, 32)*(laneId >= 16);
        //     // tmpVal  += __shfl_down_sync(tmpVal, 16);
		// 	// tmpVal  += __shfl_down_sync(tmpVal, 8);
		// 	// tmpVal  += __shfl_down_sync(tmpVal, 4);
		// 	// tmpVal  += __shfl_down_sync(tmpVal, 2);
		// 	// tmpVal  += __shfl_down_sync(tmpVal, 1);
		// 	// tmpVal  =  __shfl_sync(tmpVal, 0);
		// 	// QTree[i] = tmpVal;
        //     if(j < n_topics) p_array[j] = p_prefix_sum;
        //     if(laneId == 31) p_array_warp_sum[j / 32 ] = p_prefix_sum;
        // }
        // __syncthreads();
        // if(warpId == 0){
        //     TValue tmp_sum = 0.0;
        //     for (TId j = laneId; j < (n_topics-1)/32 + 1;  j += 32 )
        //     {   
        //         p_prefix_sum = 0.0;             
        //         p_prefix_sum = p_array_warp_sum[j];
        //         p_prefix_sum += __shfl_up_sync(p_prefix_sum, 1, 32)*(laneId >= 1);
        //         p_prefix_sum += __shfl_up_sync(p_prefix_sum, 2, 32)*(laneId >= 2);
        //         p_prefix_sum += __shfl_up_sync(p_prefix_sum, 4, 32)*(laneId >= 4);
        //         p_prefix_sum += __shfl_up_sync(p_prefix_sum, 8, 32)*(laneId >= 8);
        //         p_prefix_sum += __shfl_up_sync(p_prefix_sum, 16, 32)*(laneId >= 16);
        //         p_array_warp_sum[j] = p_prefix_sum + tmp_sum;
        //         tmp_sum += __shfl_sync(p_prefix_sum, 31); // take care
        //     }
        // }
        // __syncthreads();
        // //Sampling
        // TValue u = 0.0;
        // if(warpId == 0){
        //     if (laneId == 0) u = curand_uniform(&(randState[i])) / 1.00001;
        //     u = __shfl_sync(u, 0);
        //     for (TId j = laneId; j < (n_topics-1)/32 + 1;  j += 32 )
        //     {

        //     }
        // }
    */
    }
}
void Para::Update_iteration(Tree& tree, cudaStream_t& stream){
    gpuErrchk(cudaStreamSynchronize(stream));
    n_topics            = tree.num_nodes + 1;
    d_path_pool         = &tree.d_path_pool[0];
	d_latent_path       = &tree.d_latent_path[0];
	d_latent_path_prob  = &tree.d_latent_path_prob[0];
    d_map_old_to_new    = &tree.d_map_old_to_new[0];
    gpuErrchk(cudaStreamSynchronize(stream));
    printf("number of topics: %d\n", n_topics);
    gpuErrchk(cudaFreeAsync(d_topic_word_count,    stream));
    gpuErrchk(cudaFreeAsync(d_topic_count,         stream));
    gpuErrchk(cudaMallocAsync(&d_topic_word_count,    sizeof(TCount)*n_topics*n_words, stream));
    gpuErrchk(cudaMallocAsync(&d_topic_count,         sizeof(TCount)*n_topics, stream));
    gpuErrchk(cudaMemsetAsync(d_topic_word_count,  0,  sizeof(TCount)*n_topics*n_words, stream));
    gpuErrchk(cudaMemsetAsync(d_topic_count,    0,     sizeof(TCount)*n_topics, stream));

}

void Para::Sample_process(cudaStream_t& stream, int iter){
    auto sync_exec_policy = thrust::cuda::par.on(stream);
    if(iter == 0){
        MemoryAlloc(stream);
        MemoryHostToDevice(stream);
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    MapNewTreeId(  d_token_node_id, 
                   d_token_node_id_tmp, 
                   d_map_old_to_new,
                   stream,
                   n_tokens);

    gpuErrchk(cudaMemsetAsync(d_doc_level_count,  0, sizeof(TCount)*n_levels*n_documents,   stream));
    gpuErrchk(cudaMemsetAsync(d_topic_word_count, 0, sizeof(TCount)*n_topics*n_words,       stream));
    gpuErrchk(cudaMemsetAsync(d_topic_count,      0, sizeof(TCount)*n_topics,               stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    Update_D_W_kernel<<<GridDim, BlockDim, 0, stream>>>(    d_doc_level_count,
                                                            d_topic_word_count,
                                                            n_words,
                                                            n_levels,
                                                            n_tokens,
                                                            n_topics,
                                                            d_token_doc_id,
                                                            d_token_word_id,
                                                            d_token_node_id,
                                                            d_token_level_id);
    gpuErrchk(cudaStreamSynchronize(stream));
    Update_S_kernel<<<GridDim, BlockDim, 0, stream>>>(  d_topic_word_count,
                                                        d_topic_count,
                                                        n_words,
                                                        n_topics);

    perp_kernel<<<GridDim, BlockDim, 0, stream>>>(  d_perp,
                                                    d_doc_level_count,
                                                    d_topic_word_count,
                                                    d_topic_count,
                                                    d_alpha,
                                                    d_beta,
                                                    d_token_doc_id,
                                                    d_token_word_id,
                                                    d_token_node_id,
                                                    d_token_level_id,
                                                    d_doc_offsets,
                                                    n_words,
                                                    n_levels,
                                                    n_tokens);
    TValue result = 0.0;      
    result = perp_sum(  d_perp,
                        n_tokens,
                        stream);
    printf("Perp for iter %d is: %f \n", iter, result);

    if(iter == 199){
        // dump_device_vector<TId>(d_token_node_id, n_tokens,  MACRO_VARIABLE_TO_STRING(d_token_node_id), iter);
        // dump_device_vector<TId>(d_token_level_id, n_tokens,  MACRO_VARIABLE_TO_STRING(d_token_level_id), iter);
        // dump_device_mat<TId>(d_doc_level_count, n_documents, n_levels, MACRO_VARIABLE_TO_STRING(d_doc_level_count), iter);
        dump_device_mat<TId>(d_topic_word_count, n_topics, n_words, MACRO_VARIABLE_TO_STRING(d_topic_word_count), iter);
        dump_device_vector<TId>(d_topic_count, n_topics, MACRO_VARIABLE_TO_STRING(d_topic_count), iter);
    }

    gpuErrchk(cudaStreamSynchronize(stream));
    initRandState << <GridDim, BlockDim, 0, stream >> >(rand_u);
    SampleZ_kernel<<<GridDim, BlockDim, 0, stream>>>(   d_doc_level_count,
                                                        d_alpha,
                                                        d_topic_word_count,
                                                        d_beta,
                                                        d_topic_count,
                                                        n_words,
                                                        n_levels,
                                                        n_tokens,
                                                        n_documents,
                                                        d_token_doc_id,
                                                        d_token_word_id,
                                                        d_token_node_id,
                                                        d_token_level_id,
                                                        d_path_pool,
                                                        rand_u);

    gpuErrchk(cudaMemsetAsync(d_doc_level_count,  0, sizeof(TCount)*n_levels*n_documents,   stream));
    gpuErrchk(cudaMemsetAsync(d_topic_word_count, 0, sizeof(TCount)*n_topics*n_words,       stream));
    gpuErrchk(cudaMemsetAsync(d_topic_count,      0, sizeof(TCount)*n_topics,               stream));

    Update_D_W_kernel<<<GridDim, BlockDim, 0, stream>>>(    d_doc_level_count,
                                                            d_topic_word_count,
                                                            n_words,
                                                            n_levels,
                                                            n_tokens,
                                                            n_topics,
                                                            d_token_doc_id,
                                                            d_token_word_id,
                                                            d_token_node_id,
                                                            d_token_level_id);

    Update_S_kernel<<<GridDim, BlockDim, 0, stream>>>(  d_topic_word_count,
                                                        d_topic_count,
                                                        n_words,
                                                        n_topics);

    Update_C_M( d_token_doc_id,
                d_token_word_id, // wd
                d_token_node_id, 
                d_token_level_id, // zd
                d_token_exclusive_scan,
                d_token_exclusive_scan_count,
                d_topic_word_count,
                d_topic_count,
                stream,
                n_tokens,
                n_topics,
                n_words);

    gpuErrchk(cudaStreamSynchronize(stream));

    // dump_device_vector<TCount>(d_token_doc_id, n_tokens, MACRO_VARIABLE_TO_STRING(d_token_doc_id), iter);
    // dump_device_vector<TCount>(d_token_word_id, n_tokens, MACRO_VARIABLE_TO_STRING(d_token_word_id), iter);
    // dump_device_vector<TCount>(d_token_node_id, n_tokens, MACRO_VARIABLE_TO_STRING(d_token_node_id), iter);
    // dump_device_vector<TCount>(d_token_level_id, n_tokens, MACRO_VARIABLE_TO_STRING(d_token_level_id), iter);
    // dump_device_vector<TCount>(d_token_exclusive_scan, n_tokens, MACRO_VARIABLE_TO_STRING(d_token_exclusive_scan), iter);
    // dump_device_vector<TCount>(d_token_exclusive_scan_count, n_tokens, MACRO_VARIABLE_TO_STRING(d_token_exclusive_scan_count), iter);

    int size_shared_memory = ((n_topics-1)/BlockDim + 1) * BlockDim * sizeof(TValue) + BlockDim * sizeof(TValue) + 2 * sizeof(TValue) + 1 * sizeof(TId);
    SampleC_kernel<<<GridDim, BlockDim, size_shared_memory, stream>>>(  d_token_doc_id,
                                                                        d_token_word_id, // wd
                                                                        d_token_node_id,
                                                                        d_token_level_id, // zd
                                                                        d_token_exclusive_scan,
                                                                        d_token_exclusive_scan_count,
                                                                        d_path_pool,
                                                                        d_latent_path,
                                                                        d_latent_path_prob,
                                                                        d_topic_word_count, // b_tv
                                                                        d_beta,
                                                                        d_topic_count,
                                                                        d_doc_offsets,
                                                                        n_topics,
                                                                        n_words,
                                                                        n_levels,
                                                                        n_documents,
                                                                        rand_u);
    // gpuErrchk(cudaMemsetAsync(d_topic_word_count,  0, sizeof(TCount)*n_topics*n_words,   stream));
    // Update_W_kernel<<<GridDim, BlockDim, 0, stream>>>(d_token_word_id, d_token_node_id, d_topic_word_count, n_tokens, n_topics, n_words);
    
    
    gpuErrchk(cudaStreamSynchronize(stream));
    // if(iter/20==0){
    //     dump_device_mat<TId>(d_path_pool, L, n_documents, MACRO_VARIABLE_TO_STRING(d_path_pool), iter);
    // }

}






// // void UpdateMat_kernel()

// void UpdateMat_kernel(TCount* doc_level_count,
//                       TValue* alpha,
//                       TCount* topic_word_count,
//                       TValue* beta,
//                       TValue* topic_count,
//                       TCount  n_words,
//                       TCount  n_levels,
//                       TCount  n_tokens,
//                       TId*    token_doc_id,
//                       TId*    token_word_id, // wd
//                       TId*    token_node_id, 
//                       TId*    token_level_id, // zd
//                       TId*    token_set_count,
//                       TId*    token_map,
//                       TId*    token_map_out,
//                       TId*    doc_path,
//                       TCount* doc_offsets,
//                       float*  rand_u,
//                       cudaStream_t& stream)
// {

//     // void     *d_temp_storage = NULL;
//     // size_t   temp_storage_bytes = 0;

//     // segment sort token_word_id
//     cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,
//                                             temp_storage_bytes,
//                                             token_word_id,
//                                             token_word_id_out,
//                                             token_map,
//                                             token_map_out,
//                                             n_tokens,
//                                             n_documents,
//                                             doc_offsets,
//                                             doc_offsets + 1,
//                                             0,
//                                             sizeof(TId)*8,
//                                             stream);

//     cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);

//     cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,
//                                             temp_storage_bytes,
//                                             token_word_id,
//                                             token_word_id_out,
//                                             token_map,
//                                             token_map_out,
//                                             n_tokens,
//                                             n_documents,
//                                             doc_offsets,
//                                             doc_offsets + 1,
//                                             0,
//                                             sizeof(TId)*8,
//                                             stream);

//     auto sync_exec_policy = thrust::cuda::par.on(stream);
//     thrust::device_ptr<TId> ptr_token_level_id(token_level_id);
//     thrust::device_ptr<TId> ptr_token_map_out(token_map_out);
//     thrust::device_ptr<TId> ptr_token_level_id_out(token_level_id_out);

//     thrust::scatter(sync_exec_policy, ptr_token_level_id, ptr_token_level_id + n_tokens, ptr_token_map_out, ptr_token_level_id_out);
//     // segment sort token_level_id

//     cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,
//                                             temp_storage_bytes,
//                                             token_level_id_out,
//                                             token_level_id,// final results
//                                             token_map_out,
//                                             token_map,// final map
//                                             n_tokens,
//                                             n_documents,
//                                             doc_offsets,
//                                             doc_offsets + 1,
//                                             0,
//                                             sizeof(TId)*8,
//                                             stream);

//     cudaFreeAsync(&d_temp_storage, stream);

//     thrust::device_ptr<TId> ptr_token_word_id(token_word_id);
//     thrust::device_ptr<TId> ptr_token_map(token_map);
//     thrust::device_ptr<TId> ptr_token_word_id_out(token_word_id_out);
//     thrust::scatter(sync_exec_policy, ptr_token_word_id, ptr_token_word_id + n_tokens, ptr_token_map, ptr_token_word_id_out);
//     gpuErrchk(cudaMemcpyAsync(token_word_id, token_word_id_out, sizeof(TId)*n_tokens, cudaMemcpyDeviceToDevice, stream));

//     GenerateCount_kernel<<<256, 256, 0, stream>>>(token_set_count, doc_offsets, token_word_id, token_level_id);


//     // thrust::scatter(sync_exec_policy, ptr_token_word_id, ptr_token_word_id + n_tokens, ptr_token_map, ptr_token_word_id_out);




// ///




// // scatter the results by the new map



//       //-----------------stable sort the pool vector start-------------------
//     thrust::device_ptr<TId> ptr_0(path_pool.device_data);
//     thrust::device_ptr<TId> ptr_1(path_pool.device_data + corpus.D);
//     thrust::device_ptr<TId> ptr_2(path_pool.device_data + 2*corpus.D);



//     // cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);

//     // cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,
//     //                                         temp_storage_bytes,
//     //                                         token_level_id,
//     //                                         token_level_id_out,
//     //                                         token_map_out,
//     //                                         token_map,
//     //                                         n_tokens,
//     //                                         n_documents,
//     //                                         doc_offsets,
//     //                                         doc_offsets + 1,
//     //                                         0,
//     //                                         sizeof(TId)*8,
//     //                                         stream);

//     // auto sync_exec_policy = thrust::cuda::par.on(stream);
//     // // calculate new offset
//     // // thrust::device_ptr<TId> 
//     // token_level_id_out





//     cudaMallocAsync(&d_child_ids,  sizeof(TId)*num_nodes, stream);
//     cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,
//                                             temp_storage_bytes,
//                                             d_keys_in,
//                                             d_keys_out,
//                                             d_values_in,
//                                             d_values_out,
//                                             num_items,
//                                             num_segments,
//                                             d_offsets,
//                                             d_offsets + 1,
//                                             stream);
// }


// __global__ void SampleP_kernel( TCount* topic_word_count,
//                                 TId*    token_doc_id,
//                                 TId*    token_word_id,
//                                 TId*    token_node_id,
//                                 TId*    token_level_id,
//                                 TCount* doc_offsets,
// )
// {




// }








// void Para::SampleZ(cudaStream_t& stream){
    
    

// }

