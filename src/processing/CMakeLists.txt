
set(HEADER_LIST "${CuFront_SOURCE_DIR}/include/cufronts/processing/processing_steps.cuh")

###########################
# - A Library of all the headers
add_library(processing_library_interface INTERFACE)

###################################################################################################
# - define target   -------------------------------------------------------------------------------

target_include_directories(processing_library_interface INTERFACE 
                            "${cudahelpers_SOURCE_DIR}" ,
                            "../../include/" ,
                            "${rmm_SOURCE_DIR}/include/" ,
                            "${spdlog_SOURCE_DIR}/include/" ,
                            "${argparse_SOURCE_DIR}/include/" )


target_link_libraries(processing_library_interface INTERFACE helpers)
target_link_libraries(processing_library_interface INTERFACE spdlog)
target_link_libraries(processing_library_interface INTERFACE IO_library_interface)
target_link_libraries(processing_library_interface INTERFACE types_library_interface)

add_library(processing_library processing_steps.cu)
target_link_libraries(processing_library PUBLIC processing_library_interface)

target_compile_features(processing_library PUBLIC cxx_std_17)
target_compile_options(processing_library PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)

