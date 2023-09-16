// #include "matrix.h"

// Matrix::Matrix(TCount n_rows, TCount n_cols, bool h_flag, bool d_flag, cudaStream_t& stream) {
//     num_rows = n_rows;
//     num_cols = n_cols;
//     device_flag = d_flag;
//     host_flag = h_flag;
//     if(host_flag){
//         host_data.resize(num_rows*num_cols, 0);
//     }
//     if(device_flag){
//         gpuErrchk(cudaMallocAsync(&device_data,    sizeof(TCount)*num_rows*num_cols, stream));
//         // cudaMallocAsync((void**)&device_data, sizeof(TCount)*num_rows*num_cols, stream);
//     }

// }