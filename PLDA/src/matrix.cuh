#ifndef __MATRIX_H
#define __MATRIX_H

#include <vector>
#include <algorithm>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "type.h"
#include "util.h"
#include "corpus.h"
// #include <helper_cuda.h>
// #include <helper_functions.h>
template<class T>
class Matrix {
public:
    Matrix(TCount n_rows, TCount n_cols, bool h_flag, bool d_flag, cudaStream_t& stream)
    {
        num_rows = n_rows;
        num_cols = n_cols;
        device_flag = d_flag;
        host_flag = h_flag;
        if(host_flag){
            host_data.resize(num_rows*num_cols, 0);
        }
        if(device_flag){
            gpuErrchk(cudaMallocAsync(&device_data,    sizeof(T)*num_rows*num_cols, stream));
        }

    }
    void init_with_corpus(Corpus &cor)
    {
        for (int i=0; i<cor.D; i++){
            // std::cout<<cor.D<<std::endl;
            for(int j=0; j<L; j++){
                host_data[j*cor.D + i] = cor.corpus[i].path[j];
                // std::cout<<cor.corpus[i].path[j]<<std::endl;
            }
        }
    }
    // void set_row(int new_R) {
    //     data_resize(new_R, C);
    // }

    // void set_col(int new_C) {
    //     data_resize(R, new_C);
    // }

    // int GetR() { return R; }

    // int GetC() { return C; }

    // void data_resize(int new_n_rows, int new_n_cols) {
    //     if (new_n_rows > num_rows || new_n_cols> num_cols) {
    //         int old_n_rows = num_rows;
    //         int old_n_cols = num_cols;

    //         while (R < new_R) R = R * 2 + 1;
    //         while (C < new_C) C = C * 2 + 1;

    //         std::vector<T> old_data = std::move(host_data);

    //         data.resize(R * C);
    //         fill(data.begin(), data.end(), 0);

    //         for (int r = 0; r < old_R; r++)
    //             copy(old_data.begin() + r * old_C, old_data.begin() + (r + 1) * old_C,
    //                  data.begin() + r * C);
    //     }
    // }

    // void PermuteColumns(std::vector<int> permutation) {
    //     Matrix original = *this;

    //     fill(data.begin(), data.end(), 0);
    //     for (int c = 0; c < (int) permutation.size(); c++)
    //         if (permutation[c] != -1) {
    //             int dest = permutation[c];
    //             for (int r = 0; r < R; r++)
    //                 (*this)(r, dest) = original(r, c);
    //         }
    // }

    void host_to_device(cudaStream_t& stream) {
        // cudaMallocAsync(&device_data, 1, stream);
        cudaMemcpyAsync(device_data, host_data.data(), sizeof(T)*num_rows*num_cols, cudaMemcpyHostToDevice, stream);
    }

    void device_to_host(cudaStream_t& stream) {
        // cudaMallocAsync(&device_data, 1, stream);
        cudaMemcpyAsync(host_data.data(), device_data, sizeof(T)*num_rows*num_cols, cudaMemcpyDeviceToHost, stream);
    }

    // T &operator()(int r, int c) {
    //     return data[r * C + c];
    // }

    // T *RowPtr(int r) {
    //     return &data[r * C];
    // }

    // T *Data() {
    //     return data.data();
    // }

    // void Clear() {
    //     memset(data.data(), 0, sizeof(T) * R * C);
    // }


    TCount          num_rows;
    TCount          num_cols;
    bool            host_flag;
    bool            device_flag;
    std::vector<T>  host_data;
    T*              device_data;
};


// Matrix<TCount> MatrixInt;





#endif