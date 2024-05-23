#pragma once
#include "conversion_helper.hpp"
#include "cuda_helpers.cuh"
namespace conversion{

    template<class T>
    GLOBALQUALIFIER
    void calculate_equivalent_potential_temperature_grid_gpu(fronts::atmosphericSpatialGrid<T> temperatures, fronts::atmosphericSpatialGrid<T> specific_humidities, fronts::atmosphericSpatialGrid<T> relative_humidities, T* pressures, fronts::atmosphericSpatialGrid<T> equivalent_potential_temperature_buffer){
        auto myPos = helpers::global_thread_3D_pos_z_fastest();
        if(helpers::is_in(myPos, dim3(0,0,0), dim3(temperatures.extent(0),temperatures.extent(1),temperatures.extent(2)))){
            const T pressure = pressures[myPos.x];
            const T temperature = temperatures(myPos.x, myPos.y, myPos.z);
            const T specific_humidity = specific_humidities(myPos.x, myPos.y, myPos.z);
            const T relative_humidity = relative_humidities(myPos.x, myPos.y, myPos.z);
            equivalent_potential_temperature_buffer(myPos.x, myPos.y, myPos.z) = equivalent_potential_temperature(temperature,specific_humidity,relative_humidity,pressure);
        }
    }

    template<class device_container_t>
    void calculate_equivalent_potential_temperature_grid_gpu(device_container_t& temperatures, device_container_t& specific_humidities, device_container_t& relative_humidities, device_container_t& pressures, device_container_t& equivalent_potential_temperature_buffer, val3<size_t> dims, cudaStream_t& stream, int maxThreadsInABlock){
        fronts::atmosphericSpatialGrid<typename device_container_t::value_type> temperature_grid(temperatures.data(), dims.x,dims.y,dims.z);
        fronts::atmosphericSpatialGrid<typename device_container_t::value_type> specific_humidity_grid(specific_humidities.data(), dims.x,dims.y,dims.z);
        fronts::atmosphericSpatialGrid<typename device_container_t::value_type> relative_humidity_grid(relative_humidities.data(), dims.x,dims.y,dims.z);
        fronts::atmosphericSpatialGrid<typename device_container_t::value_type> equivalent_potential_temperature_grid(equivalent_potential_temperature_buffer.data(), dims.x,dims.y,dims.z);
        dim3 threads(32,16,maxThreadsInABlock/(32*16));
        dim3 blocks((dims.z+threads.x-1)/threads.x,(dims.y+threads.y-1)/threads.y,(dims.x+threads.z-1)/threads.z );
        calculate_equivalent_potential_temperature_grid_gpu<<<blocks,threads,0,stream>>>(temperature_grid, specific_humidity_grid, relative_humidity_grid,thrust::raw_pointer_cast(pressures.data()), equivalent_potential_temperature_grid);
    }


    template<class T>
    GLOBALQUALIFIER
    void calculate_baroclinity_gpu(fronts::atmosphericSpatialGrid<T> geopotential_buffer, fronts::atmosphericSpatialGrid<T> equivalent_potential_temperature_buffer, fronts::atmosphericSpatialGrid<T> baroclinity_buffer){
        auto myPos = helpers::global_thread_3D_pos_z_fastest();
	    // dims = level, lat, lon
        if(helpers::is_in(myPos, dim3(0,0,0), dim3(geopotential_buffer.extent(0),geopotential_buffer.extent(1),geopotential_buffer.extent(2)))){
            if(helpers::is_in(myPos, dim3(0,1,1), dim3(geopotential_buffer.extent(0),geopotential_buffer.extent(1)-1,geopotential_buffer.extent(2)-1))){
                float myLat = (90-myPos.y*0.25)/180*M_PI;
                float latSize = 6371*2*M_PI/1440.0f;
                float lonSize = cos(myLat)*latSize;
                const T geopotential_grad_x = (geopotential_buffer(myPos.x, myPos.y+1, myPos.z)                     - geopotential_buffer(myPos.x, myPos.y-1, myPos.z))/(2*latSize);
                const T ept_grad_x =          (equivalent_potential_temperature_buffer(myPos.x, myPos.y+1, myPos.z) - equivalent_potential_temperature_buffer(myPos.x, myPos.y-1, myPos.z))/(2*latSize);
                const T geopotential_grad_y = (geopotential_buffer(myPos.x, myPos.y, myPos.z+1)                     - geopotential_buffer(myPos.x, myPos.y, myPos.z-1))/(2*lonSize);
                const T ept_grad_y =          (equivalent_potential_temperature_buffer(myPos.x, myPos.y, myPos.z+1) - equivalent_potential_temperature_buffer(myPos.x, myPos.y, myPos.z-1))/(2*lonSize);
                baroclinity_buffer(myPos.x, myPos.y, myPos.z) = geopotential_grad_x*ept_grad_y - geopotential_grad_y*ept_grad_x;
            }
	    else 
                baroclinity_buffer(myPos.x, myPos.y, myPos.z) = 0;
	}
    }

    template<class device_container_t>
    void calculate_baroclinity_grid_gpu(device_container_t& geopotential_buffer, device_container_t& equivalent_potential_temperature_buffer, device_container_t& baroclinity_buffer, val3<size_t> dims, cudaStream_t& stream){
        fronts::atmosphericSpatialGrid<typename device_container_t::value_type> geopotential_grid(geopotential_buffer.data(), dims.x,dims.y,dims.z);
        fronts::atmosphericSpatialGrid<typename device_container_t::value_type> equivalent_potential_temperature_grid(equivalent_potential_temperature_buffer.data(), dims.x,dims.y,dims.z);
        fronts::atmosphericSpatialGrid<typename device_container_t::value_type> baroclinity_grid(baroclinity_buffer.data(), dims.x,dims.y,dims.z);
        dim3 threads(32,16,2);
        dim3 blocks((dims.z+dims.z-1)/threads.x,(dims.y+dims.y-1)/threads.y,(dims.x+dims.x-1)/threads.z );
        calculate_baroclinity_gpu<<<blocks,threads,0,stream>>>(geopotential_grid, equivalent_potential_temperature_grid, baroclinity_grid);
    }

};
