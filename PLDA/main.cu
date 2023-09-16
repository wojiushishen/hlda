#include "corpus.h"
#include "tree.h"
#include "hlda.h"
#include <thrust/version.h>
using namespace std;


int main(int argc, char **argv) {
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t  setVal = UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &setVal);
    float *d_a;
    string data_path = "/home/shilong/DSLDADataSet/docword.nips.txt";
    Corpus corpus(data_path.c_str());
    cout << "corpus is loaded" << endl;
    std::vector<TId> test;
    test = init_path();
    print_data(corpus.corpus[5].path);
    cudaMallocAsync(&d_a, 1, 0);
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    std::cout << "Thrust v" << major << "." << minor << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    Tree tree(corpus, stream);
    tree.InitTreeFromCorpus(corpus,stream);
    tree.CalculateProb(stream, 0);
    Para hlda_para(corpus, tree, stream);
    for (int iter = 0; iter <200; iter++)
    {
        printf("Iteration: %d\n", iter);
        hlda_para.Sample_process(stream, iter);
        tree.MemFree(stream);
        tree.CalculateProb(stream, iter+1);
        hlda_para.Update_iteration(tree, stream);
    }
}

// #include <thrust/gather.h>
// #include <thrust/device_vector.h>
// #include <thrust/execution_policy.h>

// int main(int argc, char **argv) {
//     // mark even indices with a 1; odd indices with a 0
//     int values[10] = {1, 0, 1, 0, 1, 0, 1, 5, 3, 0};
//     thrust::device_vector<int> d_values(values, values + 10);

//     // gather all even indices into the first half of the range
//     // and odd indices to the last half of the range
//     int map[12]   = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 8, 7};
//     thrust::device_vector<int> d_map(map, map + 12);

//     thrust::device_vector<int> d_output(12);
//     thrust::gather(thrust::device,
//                 d_map.begin(), d_map.end(),
//                 d_values.begin(),
//                 d_output.begin());

//     for(int i = 0; i < d_output.size(); i++)
//         std::cout << "D[" << i << "] = " << d_output[i] << std::endl;
// }