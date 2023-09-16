#ifndef __UTIL_H
#define __UTIL_H

#include "type.h"
#include "const.h"
#include <sys/stat.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


// CUDA ERROR CHECKING CODE


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) getchar();
   }
}

template<class T>
void dump_data(std::vector<T>& data_array, const char *file_name) {
    struct stat sb;
    if (stat(dump_dir.c_str(), &sb) != 0)
    {   
        mkdir(dump_dir.c_str(),0777);
        std::cout << "The Path is invalid!";
    }
    std::ofstream data_out((dump_dir + std::string(file_name)).c_str(), std::ios::binary);
    for (int i = 0; i < data_array.size(); i++) {
		data_out << data_array[i] << "\n";
	}
    data_out.close();
}


template<class T>
void dump_data_ptr(T* data_array, int size, const char *file_name, const int iter) {
    struct stat sb;
    if (stat(dump_dir.c_str(), &sb) != 0)
    {   
        mkdir(dump_dir.c_str(),0777);
        std::cout << "The Path is invalid!";
    }
    // struct stat sb2;
    // if (stat(dump_dir.c_str() + std::to_string(iter), &sb2) != 0)
    // {   
    //     mkdir(dump_dir.c_str() + std::to_string(iter), 0777);
    //     std::cout << "folder" << iter << "is created!";
    // }

    std::ofstream data_out((dump_dir + std::string(file_name)+ std::to_string(iter)).c_str(), std::ios::binary);
    for (int i = 0; i < size; i++) {
		data_out << data_array[i] << "\n";
	}
    data_out.close();
}

template<class T>
void dump_mat_ptr(T* data_array, int num_rows, int num_cols, const char *file_name, const int iter) {
    struct stat sb;
    if (stat(dump_dir.c_str(), &sb) != 0)
    {   
        mkdir(dump_dir.c_str(),0777);
        std::cout << "The Path is invalid!";
    }
    // struct stat sb2;
    // if (stat(dump_dir.c_str() + std::to_string(iter), &sb2) != 0)
    // {   
    //     mkdir(dump_dir.c_str() + std::to_string(iter), 0777);
    //     std::cout << "folder" << iter << "is created!";
    // }

    std::ofstream data_out((dump_dir + std::string(file_name) + std::to_string(iter)).c_str(), std::ios::binary);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++){
		    data_out << data_array[i*num_cols + j] << "\t";
        }
        data_out << "\n";
	}
    data_out.close();
}


template<class T>
void dump_pool(std::vector<T>& data, int num_rows, int num_cols, const char *file_name) {
    struct stat sb;
    if (stat(dump_dir.c_str(), &sb) != 0)
    {   
        mkdir(dump_dir.c_str(),0777);
        std::cout << "The Path is invalid!";
    }
    std::ofstream data_out((dump_dir + std::string(file_name)).c_str(), std::ios::binary);
    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_rows; j++){
		    data_out << data[j*num_cols + i] << " ";
        }
        data_out << "\n";
	}
    data_out.close();
}





std::vector<TId> init_path();

template<class T>
void print_data(std::vector<T>& data_array){
    for (int i = 0; i < data_array.size(); i++)
    {
        std::cout << data_array[i] << std::endl;
    }
}


template<class T> void dump_device_vector(T* out_pointer, int size, const char *file_name, int iter){
    thrust::device_vector<T> out_device(out_pointer, out_pointer + size);
    thrust::host_vector<T> out_host = out_device;
    dump_data_ptr<T>(out_host.data(), size, file_name, iter);
}



template<class T> void dump_device_mat(T* out_pointer, int num_rows, int num_cols, const char *file_name, int iter){
    thrust::device_vector<T> out_device(out_pointer, out_pointer + num_rows*num_cols);
    thrust::host_vector<T> out_host = out_device;
    dump_mat_ptr<T>(out_host.data(), num_rows, num_cols, file_name, iter);
}

void init_tree();







#endif
