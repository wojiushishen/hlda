
#include "util.h"


std::vector<TId> init_path(){

    // std::vector<std::vector<TId>> path_pool;
    std::vector<TId> path;
    std::vector<TCount> offset;
    TCount temp_offset = 0;
    TId temp_rank_id= 0;
    const int log_initial_child_counts = std::log2(initial_child_counts);
    for (int i=0; i<L; i++){
        offset.push_back(temp_offset);
        temp_offset += std::pow(initial_child_counts,i+1);
    }
    // path.push_back(1);
    // int f = time(NULL)+rand();
    // std::cout<< f <<std::endl;
    // srand(f);
    // int random = rand();

    // // print seed and random number
    // std::cout << "Seed = " << time(0) << std::endl;
    // std::cout << "Random number = " << random << std::endl;


    for (int i=0; i<L; i++){
        int random = rand();
        TId rank_id = (random % (initial_child_counts));
        // std::cout << "Random number = " << rank_id << std::endl;
        temp_rank_id = (temp_rank_id << log_initial_child_counts) + rank_id;

        // std::cout << "temp_rank_id = " << log_initial_child_counts<< std::endl;
        // temp_node_id += node_id * initial_child_counts^(i-1);
        // std::vector<TId> path;

        path.push_back(temp_rank_id + 1);
    }
    for (int i=0; i<L; i++){
        path[i] += offset[i];
        // std::cout << "offset = " << offset[i-1] << std::endl;
    }
    return path;
}

