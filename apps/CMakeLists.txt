add_executable(CuFront main.cu)

target_compile_features(CuFront PRIVATE cxx_std_17)
target_compile_options(CuFront PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)
target_link_libraries(CuFront PRIVATE cufronts_library)


if(MPI_FOUND)
    add_executable(CuFrontMPI mainMPI.cu)

    target_compile_features(CuFrontMPI PRIVATE cxx_std_17)
    target_compile_options(CuFrontMPI PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)
    target_include_directories(CuFrontMPI PUBLIC ${MPI_INCLUDE_PATH})
    target_link_libraries(CuFrontMPI PUBLIC MPI::MPI_CXX)
    target_link_libraries(CuFrontMPI PRIVATE cufronts_library)
endif()