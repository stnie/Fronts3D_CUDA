#pragma once
#include <cmath>
#include "cuda_helpers.cuh"


#ifdef __CUDACC__

    #define ALIGN(x)  __align__(x)

#else

    #if defined(_MSC_VER) && (_MSC_VER >= 1300)

        // Visual C++ .NET and later

        #define ALIGN(x) __declspec(align(x))

    #else

        #if defined(__GNUC__)

        // GCC

        #define ALIGN(x)  __attribute__ ((aligned (x)))

        #else

            #define ALIGN(x)
        #endif
    #endif
#endif



// Dim 2,3,4 structs for easier usage than floatX
template<class T>
struct val3{
    T x;
    T y;
    T z;
    HOSTDEVICEQUALIFIER
    val3(T x_in = T{0}, T y_in = T{0}, T z_in = T{0})
    : x(x_in), y(y_in), z(z_in)
    {}
};

template<class T>
struct ALIGN(2*sizeof(T)) coordinates;

template<class T>
struct ALIGN(2*sizeof(T)) val2{
    T x;
    T y;
    HOSTDEVICEQUALIFIER
    val2(T x_in = T{0}, T y_in = T{0})
    : x(x_in), y(y_in){}

    template<class T2>
    HOSTDEVICEQUALIFIER
    val2(coordinates<T2> in)
    : x(in.lon), y(in.lat){}
    T lat(){
        return y;
    }
    T lon(){
        return x;
    }
    
};

template<class T>
struct ALIGN(2*sizeof(T)) coordinates{
    T lon;
    T lat;
    HOSTDEVICEQUALIFIER
    coordinates(T lon_in = T{0}, T lat_in = T{0})
    : lon(lon_in), lat(lat_in){}

    template<class T2>
    HOSTDEVICEQUALIFIER
    coordinates(val2<T2> in)
    : lon(in.x), lat(in.y){}
};

template<class T>
struct ALIGN(4*sizeof(T)) val4{
    T x;
    T y;
    T z;
    T w;
    HOSTDEVICEQUALIFIER
    val4(T x_in = T{0}, T y_in = T{0}, T z_in = T{0}, T w_in = T{0})
    : x(x_in), y(y_in), z(z_in), w(w_in)
    {}
};