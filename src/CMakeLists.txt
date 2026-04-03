target_include_directories(cusar
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cusar
    PUBLIC
)

set_target_properties(cusar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
