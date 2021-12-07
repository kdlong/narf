#ifndef HIST_HELPERS
#define HIST_HELPERS

#include <numeric>
#include <ROOT/RVec.hxx>

template <typename T>
ROOT::VecOps::RVec<int> indices(const ROOT::VecOps::RVec<T>& vec, const int start = 0) {
    ROOT::VecOps::RVec<int> res(vec.size(), 0);
    std::iota(std::begin(res), std::end(res), start);
    return res;
}

ROOT::VecOps::RVec<int> indices(const size_t size, const int start = 0) {
    ROOT::VecOps::RVec<int> res(size, 0);
    std::iota(std::begin(res), std::end(res), start);
    return res;
}

#endif
