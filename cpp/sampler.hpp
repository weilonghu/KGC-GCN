#ifndef SAMPLER_H
#define SAMPLER_H

#include <vector>
#include "utility.hpp"

using namespace std;

// global graph data
int num_node = 0;
int num_edge = 0;
vector<int> degrees; // shape=[N]
vector<vector<vector<int>>> adj_list; // shape=[N, variable_size, 2]


void build_graph(int* src, int* rel, int* dst, int num_node_m, int num_edge_m) {
    num_node = num_node_m;
    num_edge = num_edge_m;

    // resize the vectors
    degrees.resize(num_node);
    adj_list.resize(num_node);

    for (int i = 0; i < num_edge; i++) {
        int s = src[i];
        int r = rel[i];
        int d = dst[i];

        vector<int> p = {i, d};
        vector<int> q = {i, s};
        adj_list[s].push_back(p);
        adj_list[d].push_back(q);
    }

    for (int i = 0; i < num_node; i++) {
        degrees[i] = adj_list[i].size();
    }
}


/**
 * Sample edges to form a connected graph according to
 * the adj_list and degrees of each node.
 */
void sample_edges(int sample_size, int* result){

    // Initailize
    vector<int> sample_counts(degrees);
    vector<int> picked(num_edge, 0);
    vector<int> seen(num_node, 0);

    for(int i = 0; i < sample_size; i++) {
        vector<int> weights(num_node, 0);
        for(int j = 0; j < weights.size(); j++) {
            weights[j] = sample_counts[j] * seen[j];
        }

        int sum = array_sum(weights);
        if(sum == 0){
            for(int j = 0; j < sample_counts.size(); j++) {
                if(sample_counts[j] == 0) {
                    weights[j] = 0;
                }
                else{
                    weights[j] = 1;
                }
            }
        }

        sum = array_sum(weights);
        vector<float> probabilities(weights.size(), 0);
        for (int j = 0; j < probabilities.size(); j++) {
            probabilities[j] = 1.0 * weights[j] / sum;
        }

        int chosen_vertex = random_choice_n(arange(num_node), probabilities, 1)[0];
        auto chosen_adj_list = adj_list[chosen_vertex];
        seen[chosen_vertex] = 1;

        int chosen_edge = random_choice_arange(chosen_adj_list.size());
        int edge_number = chosen_adj_list[chosen_edge][0];

        while (picked[edge_number]) {
            chosen_edge = random_choice_arange(chosen_adj_list.size());
            edge_number = chosen_adj_list[chosen_edge][0];
        }

        result[i] = edge_number;
        auto other_vetex = chosen_adj_list[chosen_edge][1];
        picked[edge_number] = 1;
        sample_counts[chosen_vertex] -= 1;
        sample_counts[other_vetex] -= 1;
        seen[other_vetex] = 1;
    }

}

#endif
