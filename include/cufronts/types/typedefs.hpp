#pragma once
#include "cufronts/types/math_helper.cuh"
#include <cuda/std/mdspan>


using DTYPE_ = float;
using PACKTYPE_ = unsigned char;

namespace fronts{

    //level, lat,lon
    template<class T>
    using atmosphericSpatialGrid = cuda::std::mdspan<T,cuda::std::dextents<size_t, 3>>;

    //sample, height, width
    template<class T>
    using crossSectionGrid = atmosphericSpatialGrid<T>;

    //sample, height
    template<class T>
    using outputGrid = cuda::std::mdspan<T,cuda::std::dextents<size_t, 2>>;
    

    // time, level, lat, lon
    template<class T>
    using atmosphericGrid = cuda::std::mdspan<T,cuda::std::dextents<size_t, 4>>;
    // time, variable, level, lat, lon
    template<class T>
    using multivariateAtmosphericGrid = cuda::std::mdspan<T,cuda::std::dextents<size_t, 5>>;
};