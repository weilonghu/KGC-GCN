#ifndef UTILITY_H
#define UTILITY_H

#include <numeric>
#include <vector>
#include <random>
#include <algorithm>

/**
 * numpy.sum
 */
template <typename T>
T array_sum(std::vector<T> array) {
    return std::accumulate(array.begin(), array.end(), 0);
}


/**
 * numpy.random.choice, choice one
 */
template <typename T>
T random_choice_one(std::vector<T> array){
    int random_index = rand() % array.size();
    return array[random_index];
}

/**
 * numpy.random.choice, given probabilities
 */
template <typename T>
std::vector<T> random_choice_n(std::vector<T> samples, std::vector<float> probabilities, int size){

    std::vector<T> vec(size);

    std::default_random_engine generator;
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

    std::vector<int> indices(vec.size());
    std::generate(indices.begin(), indices.end(), [&generator, &distribution]() { return distribution(generator); });

    std::transform(indices.begin(), indices.end(), vec.begin(), [&samples](int index) { return samples[index]; });

    return vec;
}

std::vector<int> arange(int n){
    std::vector<int> values;
    for (int i = 0; i < n; i++) {
        values.push_back(i);
    }
    return values;
}

int random_choice_arange(int n){
    auto values = arange(n);
    int random_index = rand() % values.size();
    return values[random_index];
}

#endif