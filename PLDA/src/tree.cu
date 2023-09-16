#include <cuda.h>
#include "tree.h"


// ==========================================================================================

typedef thrust::tuple<TId, TId, TId> IntTuple;



struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const {
            return a * x + y;
        }
};



// struct sort_functor
// {
//       __host__ __device__
//       bool operator()(const IntTuple& t1, const IntTuple& t2){
//             if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
//             if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;
//             if (thrust::get<1>(t1) < thrust::get<1>(t2)) return true;
//             if (thrust::get<1>(t1) > thrust::get<1>(t2)) return false;
//             if (thrust::get<2>(t1) < thrust::get<2>(t2)) return true;
//             if (thrust::get<2>(t1) > thrust::get<2>(t2)) return false;
//             return false;
//       }
// };

struct sort_functor
{
      __host__ __device__
      bool operator()(const IntTuple& t1, const IntTuple& t2){
            if (thrust::get<0>(t1) > thrust::get<0>(t2)) return true;
            if (thrust::get<0>(t1) < thrust::get<0>(t2)) return false;
            if (thrust::get<1>(t1) > thrust::get<1>(t2)) return true;
            if (thrust::get<1>(t1) < thrust::get<1>(t2)) return false;
            if (thrust::get<2>(t1) > thrust::get<2>(t2)) return true;
            if (thrust::get<2>(t1) < thrust::get<2>(t2)) return false;
            return false;
      }
};



typedef thrust::tuple<TId, TId> UniqueTuple;

struct unique_functor
{
      __host__ __device__
      bool operator()(const UniqueTuple& t1, const UniqueTuple& t2){
            if (thrust::get<0>(t1) == thrust::get<0>(t2) && thrust::get<1>(t1) == thrust::get<1>(t2)) 
            {
                  return true;
            }
            else{
                  return false;
            }
      }
};





Tree::Tree(Corpus& corpus, cudaStream_t& stream){

      // TCount* d_level_offsets;
      gpuErrchk(cudaMallocAsync(&d_level_offsets,    sizeof(TId)*(L+1), stream));
      h_extra_node_offset = new TCount[L+1];
      // // num_nodes = MAX_TOPICS;
      // num_nodes = MAX_TOPICS;
      // h_node_ids     = new TId [num_nodes];
      // h_node_levels  = new TId [num_nodes];
      // h_node_ranks   = new TId [num_nodes];
      // // h_child_ids    = new TId [];
      // gpuErrchk(cudaMallocAsync(&d_node_ids,    sizeof(TId)*num_nodes, stream));
      // gpuErrchk(cudaMallocAsync(&d_node_levels, sizeof(TId)*num_nodes, stream));
      // gpuErrchk(cudaMallocAsync(&d_node_ranks,  sizeof(TId)*num_nodes, stream));
      // //gpuErrchk(cudaMallocAsync(&d_child_ids,  ..., stream));
      
}

void Tree::TreeMemoryAlloc(cudaStream_t& stream){
      gpuErrchk(cudaMallocAsync(&d_node_ids,    sizeof(TId)*num_total_nodes, stream));
      gpuErrchk(cudaMallocAsync(&d_node_levels, sizeof(TId)*num_total_nodes, stream));
      // gpuErrchk(cudaMallocAsync(&d_node_ranks,  sizeof(TId)*num_total_nodes, stream));
      gpuErrchk(cudaMallocAsync(&d_path_count,  sizeof(TCount)*(num_nodes+1), stream));

      // gpuErrchk(cudaMallocAsync(&d_parent_node_ids,    sizeof(TId)*num_nodes, stream));
      // gpuErrchk(cudaMallocAsync(&d_child_counts,  sizeof(TCount)*num_nodes, stream));
      // gpuErrchk(cudaMallocAsync(&d_child_ids,  sizeof(TId)*num_nodes, stream));

      
      h_node_ids        = new TId[num_total_nodes];
      h_node_levels     = new TId[num_total_nodes];
      // h_node_ranks = new TId[num_total_nodes];

      // h_child_ids = new TId[num_nodes];
      h_path_count      = new TCount[num_nodes+1];
      // h_child_counts = new TCount[num_nodes];
      // h_parent_node_ids = new TId[num_nodes];

      h_latent_path = new TId[(num_nodes+1)*L];
      h_latent_path_count = new TCount[(num_nodes+1)*L];
      h_latent_path_prob = new TValue[num_nodes+1];
      gpuErrchk(cudaMallocAsync(&d_latent_path,       sizeof(TId)*(num_nodes+1)*L,        stream));
      gpuErrchk(cudaMallocAsync(&d_latent_path_count, sizeof(TCount)*(num_nodes+1)*L,     stream));
      gpuErrchk(cudaMallocAsync(&d_latent_path_prob,  sizeof(TValue)*(num_nodes+1),       stream));


      gpuErrchk(cudaMemsetAsync(d_node_ids, 0,  sizeof(TId)*num_total_nodes, stream));
      gpuErrchk(cudaMemsetAsync(d_node_levels, 0, sizeof(TId)*num_total_nodes, stream));
      gpuErrchk(cudaMemsetAsync(d_path_count, 0,  sizeof(TCount)*(num_nodes+1), stream));

      gpuErrchk(cudaMemsetAsync(d_latent_path,   0,    sizeof(TId)*(num_nodes+1)*L,        stream));
      gpuErrchk(cudaMemsetAsync(d_latent_path_count, 0, sizeof(TCount)*(num_nodes+1)*L,     stream));
      gpuErrchk(cudaMemsetAsync(d_latent_path_prob, 0,  sizeof(TValue)*(num_nodes+1),       stream));



}

void Tree::TreeMemoryFree(cudaStream_t& stream){
      gpuErrchk(cudaFreeAsync(d_node_ids,    stream));
      gpuErrchk(cudaFreeAsync(d_node_levels, stream));
      // gpuErrchk(cudaMallocFree(&d_node_ranks,  sizeof(TId)*num_total_nodes, stream));
      gpuErrchk(cudaFreeAsync(d_path_count,  stream));

      // gpuErrchk(cudaMallocAsync(&d_parent_node_ids,    sizeof(TId)*num_nodes, stream));
      // gpuErrchk(cudaMallocAsync(&d_child_counts,  sizeof(TCount)*num_nodes, stream));
      // gpuErrchk(cudaMallocAsync(&d_child_ids,  sizeof(TId)*num_nodes, stream));

      
      free(h_node_ids);
      free(h_node_levels);
      // h_node_ranks = new TId[num_total_nodes];

      // h_child_ids = new TId[num_nodes];
      free(h_path_count);
      // h_child_counts = new TCount[num_nodes];
      // h_parent_node_ids = new TId[num_nodes];

      free(h_latent_path);
      free(h_latent_path_count);
      free(h_latent_path_prob);
      gpuErrchk(cudaFreeAsync(d_latent_path,       stream));
      gpuErrchk(cudaFreeAsync(d_latent_path_count, stream));
      gpuErrchk(cudaFreeAsync(d_latent_path_prob,  stream));


}



void Tree::MapMemoryAlloc(cudaStream_t& stream){
    gpuErrchk(cudaMallocAsync(&d_map_old_to_new, sizeof(TId)* (max_node_ids+1), stream));
    gpuErrchk(cudaMallocAsync(&d_map_new_to_old, sizeof(TId)* (max_node_ids+1), stream));
    gpuErrchk(cudaMemsetAsync(d_map_old_to_new, 0, sizeof(TId)* (max_node_ids+1), stream));
    gpuErrchk(cudaMemsetAsync(d_map_new_to_old, 0, sizeof(TId)* (max_node_ids+1), stream));
}

void Tree::MapMemoryFree(cudaStream_t& stream){
    gpuErrchk(cudaFreeAsync(d_map_old_to_new, stream));
    gpuErrchk(cudaFreeAsync(d_map_new_to_old, stream));
}

void Tree::TreeMemoryDeviceToHost(cudaStream_t& stream){

      gpuErrchk(cudaMemcpyAsync(h_node_ids, d_node_ids, sizeof(TId)*num_total_nodes, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_node_levels, d_node_levels, sizeof(TId)*num_total_nodes, cudaMemcpyDeviceToHost, stream));
      // gpuErrchk(cudaMemcpyAsync(h_node_ranks, d_node_ranks, sizeof(TId)*num_total_nodes, cudaMemcpyDeviceToHost, stream));
      // gpuErrchk(cudaMemcpyAsync(h_child_ids, d_child_ids, sizeof(TId)*num_nodes, cudaMemcpyDeviceToHost, stream));
      // gpuErrchk(cudaMemcpyAsync(h_child_counts, d_child_counts, sizeof(TCount)*num_nodes, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_path_count, d_path_count, sizeof(TCount)*(num_nodes+1), cudaMemcpyDeviceToHost, stream));
      // gpuErrchk(cudaMemcpyAsync(h_parent_node_ids, d_parent_node_ids, sizeof(TId)*num_nodes, cudaMemcpyDeviceToHost, stream));

      gpuErrchk(cudaMemcpyAsync(h_latent_path, d_latent_path, sizeof(TId)*(num_nodes+1)*L, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_latent_path_count, d_latent_path_count, sizeof(TCount)*(num_nodes+1)*L, cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaMemcpyAsync(h_latent_path_prob, d_latent_path_prob, sizeof(TValue)*(num_nodes+1), cudaMemcpyDeviceToHost, stream));
      // gpuErrchk(cudaMemcpyAsync(h_node_ranks, d_node_ranks, sizeof(TId)*num_nodes, cudaMemcpyDeviceToHost, stream));
}

// void Tree::TreeMemoryFree(cudaStream_t& stream){
//       gpuErrchk(cudaFreeAsync(d_node_ids, stream));
//       gpuErrchk(cudaFreeAsync(d_node_levels, stream));
//       gpuErrchk(cudaFreeAsync(d_node_ranks, stream));
// }



void GenerateLatentPath(TId*          latent_path,
                        TCount*       latent_path_count,
                        TId*          path_pool_sorted,
                        TCount        n_documents,
                        TCount        n_nodes,
                        TCount*       h_extra_node_offset,
                        TCount*       path_count,
                        cudaStream_t& stream)
{
      auto sync_exec_policy = thrust::cuda::par.on(stream);
      thrust::constant_iterator<int> const_one_first(1);
      thrust::counting_iterator<int> sequence_first(0);
      thrust::device_ptr<TId>        ptr_latent_path(latent_path);
      thrust::device_ptr<TCount>     ptr_latent_path_count(latent_path_count);
      thrust::device_ptr<TId>        ptr_path_pool_sorted(path_pool_sorted);
      thrust::device_ptr<TCount>     ptr_path_count(path_count);

      // TId* index_array =   
      // gpuErrchk(cudaMallocAsync(&tmp_node_ids,    sizeof(TId)*num_documents*L, stream));


      // sort_functor<TId> sortFunctor;
      int offset = 0;
      int offset_1 = 0;
      {
            auto in_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_path_pool_sorted, ptr_path_pool_sorted + n_documents, ptr_path_pool_sorted + 2*n_documents));
            auto in_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_path_pool_sorted + n_documents, ptr_path_pool_sorted + 2*n_documents, ptr_path_pool_sorted + 3*n_documents));
            auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_latent_path, ptr_latent_path + n_nodes, ptr_latent_path + 2*n_nodes));
            auto out_end = thrust::unique_copy(sync_exec_policy, in_begin, in_end, out_begin);
            offset += (out_end - out_begin);
      }
      {
            auto in_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_latent_path, ptr_latent_path + n_nodes));
            auto in_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_latent_path + offset, ptr_latent_path + n_nodes + offset));
            auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_latent_path + offset, ptr_latent_path + n_nodes + offset));
            auto out_end = thrust::unique_copy(sync_exec_policy, in_begin, in_end, out_begin);
            offset_1 = offset + (out_end - out_begin);
      }
      
      {
            // auto in_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_latent_path, ptr_latent_path + n_documents));
            // auto in_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_latent_path + offset));
            // auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_latent_path + offset));
            auto out_end = thrust::unique_copy(sync_exec_policy, ptr_latent_path + offset, ptr_latent_path + offset_1, ptr_latent_path + offset_1);
            offset_1 += (out_end - (ptr_latent_path + offset_1));
      }

      thrust::gather(sync_exec_policy, ptr_latent_path, ptr_latent_path + L*n_nodes, ptr_path_count, ptr_latent_path_count);

      //fill the extra nodes
      {
            for(int i=0; i<L; i++)
            {
                  thrust::sequence(sync_exec_policy, ptr_latent_path + (i+1)*n_nodes - (h_extra_node_offset[i+1] - h_extra_node_offset[i]), ptr_latent_path + (i+1)*n_nodes, h_extra_node_offset[i] + (n_nodes-1) + 1);
            }
      }
}


__global__ void GeneratePathProbLeaf(TId*     latent_path,
                                    TCount*   latent_path_count,
                                    TValue    gama, // number of element L+1, the last is never used
                                    TValue*   latent_path_prob,
                                    TCount    n_documents,
                                    TCount    n_nodes,// euqals to number of possible paths
                                    const TCount    n_levels // L
                                    )
{
      int tid = threadIdx.x + blockIdx.x * blockDim.x; //every thread process a path      

      for(int i = tid; i < n_nodes; i += blockDim.x * gridDim.x){
            bool extra_node_flag = 0;
            TValue tmp_latent_path_prob = 1.0;
            // TValue denominator = n_documents + gama[0];
            TValue denominator = n_documents + gama;
            for(int j=0; j<n_levels; j++){
                  TValue nominator = (TValue) latent_path_count[j*n_nodes + i];
                  if(nominator < 1.0){
                        extra_node_flag = 1;
                        nominator = gama;
                        // nominator = gama[j];
                  }
                  tmp_latent_path_prob = tmp_latent_path_prob * nominator/denominator;
                  denominator = nominator + gama;
                  // denominator = nominator + gama[j+1];
                  if(extra_node_flag) break;
            }
            latent_path_prob[i] = tmp_latent_path_prob;
      }
      
}


void Tree::InitTreeFromCorpus(Corpus& corpus, cudaStream_t& stream){

      num_documents = corpus.D;
      h_path_pool = new TId[num_documents * L];

      for (int i=0; i<num_documents; i++){
            for(int j=0; j<L; j++){
                  h_path_pool[j*num_documents + i] = corpus.corpus[i].path[j];
            }
      }

      cudaStreamSynchronize(stream);
      gpuErrchk(cudaMallocAsync(&d_path_pool, sizeof(TId)*num_documents*L, stream));
      gpuErrchk(cudaMallocAsync(&d_path_pool_ordered, sizeof(TId)*num_documents*L, stream));
      gpuErrchk(cudaMallocAsync(&d_path_pool_tmp, sizeof(TId)*num_documents*L, stream));
      gpuErrchk(cudaMemcpyAsync(d_path_pool, h_path_pool, sizeof(TId)*num_documents*L, cudaMemcpyHostToDevice, stream));
      

      for(int i = 0; i < 20; i++){
            std::cout << h_path_pool[i] << std::endl;
      }
      std::cout << "-------path_pool init done!--------" << std::endl;
}



void Tree::CalculateProb(cudaStream_t& stream, int iter){
      cudaStreamSynchronize(stream);
      gpuErrchk(cudaMemcpyAsync(d_path_pool_ordered, d_path_pool, sizeof(TId)*num_documents*L, cudaMemcpyDeviceToDevice, stream));
      auto sync_exec_policy = thrust::cuda::par.on(stream);
      //-----------------stable sort the pool vector start-------------------
      thrust::device_ptr<TId> path_pool_ordered_ptr(d_path_pool_ordered);

      auto begin = thrust::make_zip_iterator(thrust::make_tuple(path_pool_ordered_ptr, path_pool_ordered_ptr + num_documents, path_pool_ordered_ptr + 2*num_documents));
      auto end = thrust::make_zip_iterator(thrust::make_tuple(path_pool_ordered_ptr+ num_documents, path_pool_ordered_ptr + 2*num_documents, path_pool_ordered_ptr + 3*num_documents));

      thrust::stable_sort(sync_exec_policy, begin, end, sort_functor());
      gpuErrchk(cudaStreamSynchronize(stream));

      //-----------------------------prone the node and generate corresponding maps---------------------------

      max_node_ids = thrust::reduce(sync_exec_policy, path_pool_ordered_ptr, path_pool_ordered_ptr + 3*num_documents, -1, thrust::maximum<int>());
      std::cout << "max id is :" << max_node_ids << std::endl;
      MapMemoryAlloc(stream);
      
      thrust::device_ptr<TId> map_old_to_new_ptr(d_map_old_to_new);
      thrust::device_ptr<TId> map_new_to_old_ptr(d_map_new_to_old);

      {
            //generate map
            // auto out_end = thrust::unique_copy(sync_exec_policy, path_pool_ordered_ptr, path_pool_ordered_ptr + L*num_documents, map_new_to_old_ptr + 1);
            // thrust::counting_iterator<int> sequence_first(1);
            // thrust::scatter(sync_exec_policy, map_new_to_old_ptr + 1, out_end, sequence_first, map_old_to_new_ptr);
                        //generate map
            auto out_end = thrust::unique_copy(sync_exec_policy, path_pool_ordered_ptr, path_pool_ordered_ptr + L*num_documents, map_new_to_old_ptr + 1);
            int  n = out_end - (map_new_to_old_ptr + 1);
            thrust::counting_iterator<int> sequence_first(1);
            thrust::scatter(sync_exec_policy, sequence_first, sequence_first + n, map_new_to_old_ptr + 1, map_old_to_new_ptr);
      }
      //---------------------generate new pathpool------------------------------
      thrust::device_ptr<TId> path_pool_tmp_ptr(d_path_pool_tmp);
      thrust::device_ptr<TId> path_pool_ptr(d_path_pool);

      // thrust::device_vector<TId> outprint(path_pool_ptr, path_pool_ptr + num_documents*L);
      // thrust::host_vector<TId> out_host = outprint;
      // dump_mat_ptr<TCount>(out_host.data(), L, num_documents, MACRO_VARIABLE_TO_STRING(out_host));
      // gpuErrchk(cudaStreamSynchronize(stream));

      thrust::gather(sync_exec_policy, path_pool_ptr, path_pool_ptr + L*num_documents, map_old_to_new_ptr, path_pool_tmp_ptr);
      gpuErrchk(cudaMemcpyAsync(d_path_pool, d_path_pool_tmp, sizeof(TId)*num_documents*L, cudaMemcpyDeviceToDevice, stream));
      
      thrust::gather(sync_exec_policy, path_pool_ordered_ptr, path_pool_ordered_ptr + L*num_documents, map_old_to_new_ptr, path_pool_tmp_ptr);
      gpuErrchk(cudaMemcpyAsync(d_path_pool_ordered, d_path_pool_tmp, sizeof(TId)*num_documents*L, cudaMemcpyDeviceToDevice, stream));
      
      
      // thrust::device_vector<TId> outprint2(path_pool_ptr, path_pool_ptr + num_documents*L);
      // thrust::host_vector<TId> out_host2 = outprint2;
      // dump_mat_ptr<TCount>(out_host2.data(), L, num_documents, MACRO_VARIABLE_TO_STRING(out_host2));
      // gpuErrchk(cudaStreamSynchronize(stream));

      // for(int i = 0; i < outprint.size(); i++)
      //   std::cout << "outprint[" << i << "] = " << outprint[i] << std::endl;


      //--------------------Begin::generate node related info-------------------------------
      gpuErrchk(cudaStreamSynchronize(stream));

      thrust::device_ptr<TCount> level_offsets_ptr(d_level_offsets);
      level_offsets_ptr[0] = 0;
      
      h_extra_node_offset[0] = 0;
      h_extra_node_offset[1] = 1;
      // h_extra_node_offset[2] = 1;
      num_extra_nodes = 1;// root has one node, need 1*L extra ranks
      num_nodes = 0;
      for(int i=0; i<L; i++){            
            // thrust::sort(sync_exec_policy, ptr+i*num_documents, ptr+(i+1)*num_documents);
            auto end = thrust::unique(sync_exec_policy, path_pool_tmp_ptr + i*num_documents, path_pool_tmp_ptr + (i+1)*num_documents);
            // auto end = thrust::unique(thrust::device, path_pool.device_data + i*L, path_pool.device_data + (i+1)*L);
            cudaStreamSynchronize(stream);
            int size = end - (path_pool_tmp_ptr + i*num_documents);
            std::cout <<"length= "<< size  << std::endl;
            num_nodes += size;
            level_offsets_ptr[i+1] = num_nodes;

            if(i<(L-1)) {
                  num_extra_nodes += size;
                  h_extra_node_offset[i+2] = h_extra_node_offset[i+1] + num_extra_nodes;
            }
            
            // num_extra_nodes += size * (L - i - 1);
            // num_nodes += size;
      }
      num_extra_nodes = h_extra_node_offset[L];
      std::cout << "num_extra_nodes" << num_extra_nodes <<std::endl;
      num_total_nodes = num_extra_nodes + num_nodes;
      gpuErrchk(cudaStreamSynchronize(stream));

      TreeMemoryAlloc(stream);
      thrust::device_ptr<TId> node_ids_ptr(d_node_ids);
      thrust::device_ptr<TId> node_levels_ptr(d_node_levels);
      // thrust::device_ptr<TId> node_ranks_ptr(d_node_ranks);
      for(int i=0; i<L; i++){
            // for existing nodes
            thrust::copy(sync_exec_policy, path_pool_tmp_ptr+i*num_documents, path_pool_tmp_ptr+i*num_documents+level_offsets_ptr[i+1]-level_offsets_ptr[i], node_ids_ptr+level_offsets_ptr[i]);            
            thrust::fill(sync_exec_policy, node_levels_ptr+level_offsets_ptr[i], node_levels_ptr+level_offsets_ptr[i+1], i+1);
            // thrust::sequence(sync_exec_policy, node_ranks_ptr+level_offsets_ptr[i], node_ranks_ptr+level_offsets_ptr[i+1], 1);
      }

      thrust::sequence(sync_exec_policy, node_ids_ptr + level_offsets_ptr[L], node_ids_ptr + level_offsets_ptr[L] + h_extra_node_offset[L], num_nodes + 1);
      for(int i=0; i<L; i++){
            thrust::fill(sync_exec_policy, node_levels_ptr+level_offsets_ptr[L] + h_extra_node_offset[i], node_levels_ptr+level_offsets_ptr[L] + h_extra_node_offset[i+1], i+1);
            // thrust::sequence(sync_exec_policy, node_ranks_ptr+level_offsets_ptr[L] + h_extra_node_offset[i], node_ranks_ptr+level_offsets_ptr[L] + h_extra_node_offset[i+1], level_offsets_ptr[i+1] - level_offsets_ptr[i] + 1);
      }

      cudaStreamSynchronize(stream);



      thrust::equal_to<int> binary_pred;
      thrust::plus<int> binary_op;
      thrust::constant_iterator<int> first(1);
      // thrust::constant_iterator<int> last = first + 3;
      thrust::device_ptr<TCount> path_count_ptr(d_path_count);
      auto new_end = thrust::reduce_by_key(sync_exec_policy, path_pool_ordered_ptr, path_pool_ordered_ptr + num_documents*L, first, node_ids_ptr, path_count_ptr + 1, binary_pred, binary_op);


      gpuErrchk(cudaStreamSynchronize(stream));

      GenerateLatentPath(d_latent_path, d_latent_path_count, d_path_pool_ordered, num_documents, num_nodes + 1, h_extra_node_offset, d_path_count, stream);

      gpuErrchk(cudaStreamSynchronize(stream));
      GeneratePathProbLeaf<<<256, 256, 0, stream>>>(d_latent_path, d_latent_path_count,gama,d_latent_path_prob,num_documents,num_nodes + 1,L);
      TreeMemoryDeviceToHost(stream);
      gpuErrchk(cudaStreamSynchronize(stream));

      // // MapMemoryFree(stream);
      if(iter == 199){
            dump_mat_ptr<TId>(h_latent_path, L, num_nodes + 1, MACRO_VARIABLE_TO_STRING(h_latent_path), iter);
      }

      // dump_mat_ptr<TCount>(h_latent_path_count, L, num_nodes + 1, MACRO_VARIABLE_TO_STRING(h_latent_path_count), iter);
      // dump_data_ptr<TId>(h_node_ids, num_total_nodes, MACRO_VARIABLE_TO_STRING(h_node_ids), iter);
      // // dump_data_ptr<TId>(h_node_ranks, num_total_nodes, MACRO_VARIABLE_TO_STRING(h_node_ranks));
      // dump_data_ptr<TId>(h_node_levels, num_total_nodes, MACRO_VARIABLE_TO_STRING(h_node_levels), iter);
      // dump_data_ptr<TValue>(h_latent_path_prob, num_nodes + 1, MACRO_VARIABLE_TO_STRING(h_latent_path_prob), iter);

}

void Tree::MemFree(cudaStream_t& stream){
      MapMemoryFree(stream);
      TreeMemoryFree(stream);
}





void Tree::TreeInitialize(cudaStream_t& stream){
      
}