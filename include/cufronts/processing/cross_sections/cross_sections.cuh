#pragma once
#include <cuda_helpers.cuh>
#include <cufronts/types/math_helper.cuh>
#include "cufronts/processing/util/processingConfig.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
T evaluateTex(cudaTextureObject_t data, val3<T> pos, dim3 dims){
    T interpolated = tex2DLayered<T>(data, pos.x+0.5, pos.y+0.5, pos.z);
    return interpolated;
}
template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
T evaluateTex(cudaTextureObject_t data, val3<T> pos){
    T interpolated = tex2DLayered<T>(data, pos.x+0.5, pos.y+0.5, pos.z);
    return interpolated;
}

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
val3<T> rotateZ(val3<T> point, T angle){
    return val3<T>{
        cos(angle)*point.x - sin(angle)*point.y,
        sin(angle)*point.x + cos(angle)*point.y,
        point.z
    };
}

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
val3<T> rotateY(val3<T> point, T angle){
    return val3<T>{
        cos(angle)*point.x + sin(angle)*point.z,
        point.y,
        -sin(angle)*point.x + cos(angle)*point.z
    };
}

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
val3<T> rotateX(val3<T> point, T angle){
    return val3<T>{
        point.x,
        cos(angle)*point.y - sin(angle)*point.z ,
        sin(angle)*point.y + cos(angle)*point.z 
    };
}

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
val3<T> KartToPol(val3<T> point){
    T r = sqrt(point.x*point.x+point.y*point.y+point.z*point.z);
    return val3<T>{
        r,
        asin(point.y/r),
        atan2(point.x,point.z)
    };
}


template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
val3<T> getPosFixedStep(int level, T dir, T step_km, T latStart, T lonStart){

    const T r = 6371;
    const T cirumference = 2*M_PI*r;
    const T step_angle = step_km / r;

    const T bearing = dir;

    auto kartLatLonStart = val3<T>{0,0,r};
    // rotate around y
    auto kartLatLonEnd = rotateY(kartLatLonStart, step_angle);

    // rotate around Z
    kartLatLonEnd = rotateZ(kartLatLonEnd, bearing);
    // rotate around X
    kartLatLonEnd = rotateX(kartLatLonEnd, T(-latStart / 180 * M_PI));
    // rotate around Y
    kartLatLonEnd = rotateY(kartLatLonEnd, T( lonStart / 180 * M_PI));

    auto LatLonEnd = KartToPol(kartLatLonEnd);
    
    return val3<T>( // lon, lat, level
        (LatLonEnd.z/M_PI*180 - (-90)) / 0.25,
        (LatLonEnd.y/M_PI*180 -   90 ) /-0.25,
        level);

}


template<class T>
GLOBALQUALIFIER
void adjustOrientation(fronts::crossSectionGrid<T> crossSections, fronts::crossSectionGrid<val2<T>> locations, size_t* keys, size_t* orientations, size_t* tgt_keys, size_t* tgt_orientation, size_t* segments, T* dirs, const size_t interpolations = 16, const size_t smoothing_steps = 1){
    const auto myPos = helpers::global_thread_3D_pos_z_fastest();
    //z = width
    //y = height
    //x = samples
    if(myPos.x < crossSections.extent(0)){
        const size_t myArrayPos = myPos.x;
        const auto myKey = keys[myArrayPos];
        const auto mySegmentSize = segments[myKey+1]-segments[myKey];
        const int sampleCorrectlyOriented = orientations[myArrayPos];
        const int numberofSamplesInFront = mySegmentSize*interpolations*smoothing_steps;
        const float fractionOfCorrectlyOrientedSamplesInFront = 
            ((float)tgt_orientation[myKey])/(numberofSamplesInFront);
    
        // all fronts are initally extracted with the same orientation in respect to line orientation
        // some may have been flipped due to an underlying field.  (e.g. because fronts should be oriented along a gradient)
        // To ensure the same orientation for all samples we reorient all samples according to the majority vote
        // of whether the initial orientation was correct or not
        const bool need_flip = abs(sampleCorrectlyOriented-fractionOfCorrectlyOrientedSamplesInFront)>0.5;
	    
        
        if(need_flip){
            dim3 myOtherPos (myPos.x, myPos.y, crossSections.extent(2)-myPos.z-1);
            if(myPos.z < crossSections.extent(2)/2){
                // invalid angles write invalid values to cross section (will be ignored later on)
                const auto cs = crossSections(myPos.x,myPos.y,myPos.z);
                crossSections(myPos.x,myPos.y,myPos.z) = crossSections(myOtherPos.x,myOtherPos.y,myOtherPos.z);
                crossSections(myOtherPos.x,myOtherPos.y,myOtherPos.z) = cs;
                const auto loc = locations(myPos.x,myPos.y,myPos.z);
                locations(myPos.x,myPos.y,myPos.z) = locations(myOtherPos.x,myOtherPos.y,myOtherPos.z);
                locations(myOtherPos.x,myOtherPos.y,myOtherPos.z) = loc;
            }
            if(myPos.z==0 && myPos.y==0)
                dirs[myArrayPos] = dirs[myArrayPos]+3.141592654f;
        }
    }
}


template<class T>
GLOBALQUALIFIER
void getCSValueFromTexWOrient(fronts::crossSectionGrid<T> crossSections, fronts::crossSectionGrid<val2<T>> locations, val2<T>* bases, T* dirs, bool* needs_flip, size_t* orientations, cudaTextureObject_t data_tex, cudaTextureObject_t location_tex, cudaTextureObject_t dir_tex, const bool reverse, const float step_km, const float dir_step_km){
    auto myPos = helpers::global_thread_3D_pos_z_fastest();
    //z = width
    //y = height
    //x = sample
    dim3 outdims(crossSections.extent(0),crossSections.extent(1),crossSections.extent(2));
    if(myPos.x < outdims.x && myPos.y < outdims.y && myPos.z < outdims.z){
        bool flipDir = false;
        const size_t myArrayPos = myPos.x;

        // base is: lon, lat
        const auto base = bases[myArrayPos];
        // dir is: rad
        const auto dir = dirs[myArrayPos];
        const int myoffset = myPos.z - (outdims.z)/2;
        
        // base is: lon, lat
        const T lonAtBase = evaluateTex<T>(location_tex, val3<T>(base.x,base.y,0));
        const T latAtBase = evaluateTex<T>(location_tex, val3<T>(base.x,base.y,1));
	
        const auto myEvalPos = getPosFixedStep<T>(myPos.y, dir, myoffset*step_km, latAtBase, lonAtBase);
        
        const T myData = evaluateTex<T>(data_tex, myEvalPos);
        // craete lat / lon val2 struct
        const val2<T> myLocation(
            evaluateTex<T>(location_tex, val3<T>(myEvalPos.x,myEvalPos.y,0)),
            evaluateTex<T>(location_tex, val3<T>(myEvalPos.x,myEvalPos.y,1))
        ) ;
        
        auto myWarp = cg::tiled_partition<32>(cg::this_thread_block());
        const int myOrientationOffset = myWarp.thread_rank()<16 ? ((int)myWarp.thread_rank())-16 : myWarp.thread_rank()-15;
        const auto myOrientationEvalPos = getPosFixedStep<T>(0, dir, myOrientationOffset*dir_step_km, latAtBase, lonAtBase);

        auto myOrientationValue = evaluateTex<T>(dir_tex, myOrientationEvalPos);
        if(myWarp.thread_rank()>=16) myOrientationValue*=-1;
        const auto correctlyOriented = cg::reduce(myWarp, myOrientationValue, [](T x, T y){return x+y;}) > 0;
        
        // orient data in direction of underlying field (wind, thermal, ...)
        if(!correctlyOriented){
            // revert writing operation
            myPos.z = outdims.z-myPos.z-1;
            flipDir = !flipDir;
        }

        // reverse data if occ_2 front
        if(reverse){
            myPos.z = outdims.z-myPos.z-1;
            flipDir = !flipDir;
        }
        
        // invalid angles write invalid values to cross section (will be ignored later on)
        crossSections(myPos.x,myPos.y,myPos.z) = (dir< INVALID_ANGLE/2.0)? INVALID_DATA : myData;
        locations(myPos.x,myPos.y,myPos.z) = myLocation;
        if(myPos.y == 0 && myPos.z == 0){
            orientations[myArrayPos] = correctlyOriented ? 1: 0;
            needs_flip[myArrayPos] = flipDir;
        }
    }
}

template<class T>
GLOBALQUALIFIER
void flipDir(T* dirs, bool* needs_flip, size_t num_smpls){
    auto myPos = threadIdx.x+blockIdx.x*blockDim.x;
    if(myPos < num_smpls){
        dirs[myPos] += needs_flip[myPos]*3.14159265358979323846;
    }
}


///// GENERAL INTERPOLATION PART


template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
float getNormalDirFromLine(const val2<T>& line){
    // the normal is obtained by making one direction negative (here lineDir.y) and swapping x and y
    const val2<T> myNormal(-line.y, line.x);
    // the normal direction is now obtained by flipping the y, to get the direction in terms of latitude-oriented axis (e.g. positive direction is upwards)
    return atan2(-myNormal.y, myNormal.x);
}



template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
void getInterpolationPoint(
        const val2<T>& beginCoord, const val2<T>& endCoord, 
        const int grp_pos, const int grp_size, 
        const float overshoot, 
        const size_t rotations, 
        const size_t myThreadPos, 
        const size_t myFrontID,
        val2<T>* bases, 
        T* dirs, 
        size_t* keys)
{
    const val2<T> lineDir(endCoord.x-beginCoord.x, endCoord.y-beginCoord.y);
    const int interpolPosition = grp_pos;
    const int numInterpols = grp_size;
    float interpolFac = ((float)interpolPosition)/numInterpols;
    interpolFac -= overshoot*2;
    const val2<T> myCoord(beginCoord.x + interpolFac*lineDir.x, beginCoord.y+interpolFac*lineDir.y);
    const auto myNormalDir = getNormalDirFromLine(lineDir);

    // oversampling of angle if desired 
    for(int ang = 0; ang < rotations; ++ang){
        const auto myWritePos = ang+myThreadPos;
        bases[myWritePos] = myCoord;
        dirs[myWritePos] = myNormalDir + (ang-((int)rotations/2))*30.0/180.0*M_PI;
        keys[myWritePos] = myFrontID;
    }
}




template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
val2<T> quadraticBezierCurve(const float t, const val2<T>& p1, const val2<T>& p2, const val2<T>& p3){
    const T x = (1-t)*((1-t)*p1.x+t*p2.x)+t*((1-t)*p2.x+t*p3.x);
    const T y = (1-t)*((1-t)*p1.y+t*p2.y)+t*((1-t)*p2.y+t*p3.y);
    return val2<T>(x,y);
}

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
float quadraticBezierCurveNormalDir(const float t, const val2<T>& p1, const val2<T>& p2, const val2<T>& p3){
    const val2<T> lineDir(((1-t)*p2.x+t*p3.x) - ((1-t)*p1.x+t*p2.x), ((1-t)*p2.y+t*p3.y) - ((1-t)*p1.y+t*p2.y)); 
    return getNormalDirFromLine(lineDir);
}

template<class T>
DEVICEQUALIFIER INLINEQUALIFIER
void getInterpolationPointOnBezierCurve(
        const val2<T>& beginCoord, const val2<T>& midCoord, const val2<T>& endCoord, 
        const int grp_pos, const int grp_size, 
        const size_t rotations, 
        const size_t myThreadPos, 
        const size_t myFrontID,
        val2<T>* bases, 
        T* dirs, 
        size_t* keys)
{
    // get interpolation factor (0 to 1)
    const int interpolPosition = grp_pos;
    const int numInterpols = grp_size;
    const float interpolFac = ((float)interpolPosition)/numInterpols;

    const val2<T> myCoord = quadraticBezierCurve(interpolFac, beginCoord, midCoord, endCoord);
    const auto myNormalDir = quadraticBezierCurveNormalDir(interpolFac, beginCoord, midCoord, endCoord);

    // oversampling of angle if desired 
    for(int ang = 0; ang < rotations; ++ang){
        const auto myWritePos = ang+myThreadPos;
        bases[myWritePos] = myCoord;
        dirs[myWritePos] = myNormalDir + (ang-((int)rotations/2))*30.0/180.0*M_PI;
        keys[myWritePos] = myFrontID;
    }
}



template<class T>
GLOBALQUALIFIER 
void getPosAndDirWKey(val2<T>* coordinatePairs, size_t* offsets, size_t* segments, val2<T>* bases, T* dirs, size_t* keys, size_t* positions, const size_t num_fronts, const size_t rotations, const size_t smoothing_steps, const size_t num_interpolations){
    // assume bases is already extracted
    const auto myZPos = threadIdx.z+blockDim.z*blockIdx.z;
    
    // percentage of interpolation points that should be before and after the line segment
    // 1- 2*overshoot remains within the line segment
    const float overshoot = 0.0;    
    const auto grp_pos = threadIdx.x+blockIdx.x*blockDim.x;
    const auto grp_size = (blockDim.x*gridDim.x)*(1-2*overshoot);
    if(grp_pos < num_interpolations && myZPos < num_fronts){
        const auto myFrontID = positions[myZPos];
        // number of segments per front
        const size_t num_points_in_front = segments[myFrontID+1]-segments[myFrontID];
        // factor of how many samples are drawn from per input segment
        const size_t sampling_factor_per_smoothing_step = blockDim.x*gridDim.x*rotations;
        const size_t sampling_factor = sampling_factor_per_smoothing_step*smoothing_steps;
        
        // number of output samples per front
        const size_t num_samples_per_smoothing_step = num_points_in_front*sampling_factor_per_smoothing_step;

        // the offset for the output samples 
        const auto myZOff = segments[myFrontID]*sampling_factor;
        
        for(int point =0; point < num_points_in_front; point+= blockDim.y*gridDim.y){
            const auto myYPos = threadIdx.y+blockIdx.y*blockDim.y + point;
            // position within the front[myFrontID] 
            const auto myPosWithinFront = rotations*(threadIdx.x+blockDim.x*blockIdx.x+ myYPos*blockDim.x*gridDim.x);
            
            // position to write my output to (for the first smoothing step)
            const auto myThreadPos = myZOff + myPosWithinFront;
            if(myYPos < num_points_in_front){
                // coordinatePairs is in grid points (x = 0 == left end ==> x axis positive direction is rightwards, y= 0 == top end ==> y axis positive direction is downwards)
                // first value is the x axis (equals longitude axis), second value is the y axis (equals inverted latitude axis)) 
                const val2<T> beginCoord = coordinatePairs[offsets[myFrontID]+myYPos];
                const val2<T> endCoord = coordinatePairs[offsets[myFrontID]+myYPos+1];
                val2<T> finalCoord = coordinatePairs[offsets[myFrontID]+myYPos+1];
                getInterpolationPoint(beginCoord, endCoord, grp_pos, grp_size, overshoot, rotations, myThreadPos, myFrontID, bases, dirs, keys);
                if(smoothing_steps>1){
                    if(myYPos < num_points_in_front-1){
                        finalCoord = coordinatePairs[offsets[myFrontID]+myYPos+2];
                    }
                    getInterpolationPointOnBezierCurve(beginCoord, endCoord, finalCoord, grp_pos, grp_size, rotations, myThreadPos+num_samples_per_smoothing_step, myFrontID, bases, dirs, keys);
                }
            }
        }
    }
}





