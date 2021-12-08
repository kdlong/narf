#ifndef HIST_HELPERS
#define HIST_HELPERS

#include <numeric>
#include <ROOT/RVec.hxx>

template <typename T>
ROOT::VecOps::RVec<int> indices(const T& vec, const int start = 0) {
    ROOT::VecOps::RVec<int> res(vec.size(), 0);
    std::iota(std::begin(res), std::end(res), start);
    return res;
}

ROOT::VecOps::RVec<int> indices(const size_t size, const int start = 0) {
    ROOT::VecOps::RVec<int> res(size, 0);
    std::iota(std::begin(res), std::end(res), start);
    return res;
}

// Thanks stack overflow https://stackoverflow.com/questions/42749032/concatenating-a-sequence-of-stdarrays
template<typename T, int N, int M>
auto concatRVecs(const ROOT::VecOps::RVec<T>& vec1, ROOT::VecOps::RVec<T>& vec2)
{
    std::array<T, N+M> result;
    std::copy (vec1.cbegin(), vec1.cend(), result.begin());
    std::copy (vec2.cbegin(), vec2.cend(), result.begin() + N);
    return result;
}

#endif
