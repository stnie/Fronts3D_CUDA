
set(HEADER_LIST "${CuFront_SOURCE_DIR}/include/cufronts/processing/processing_steps.cuh")

###########################
# - A Library of all the headers
add_library(types_library_interface INTERFACE)

###################################################################################################
# - define target   -------------------------------------------------------------------------------

target_include_directories(types_library_interface INTERFACE 
                            "${cudahelpers_SOURCE_DIR}" ,
                            "../../include" )


target_link_libraries(types_library_interface INTERFACE helpers)
target_link_libraries(types_library_interface INTERFACE CCCL::CCCL)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)

