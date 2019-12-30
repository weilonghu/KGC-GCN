#ifndef LIB_H
#define LIB_H

#include "sampler.hpp"

extern "C" {
    int test(){
        int state = 1;
        return state;
    }

    void build_graph_c(int* src, int* rel, int* dst, int num_node, int num_edge) {
        build_graph(src, rel, dst, num_node, num_edge);
    }

    void sample_edges_c(int sample_size, int* result) {
        sample_edges(sample_size, result);
    }
}

#endif