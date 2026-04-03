target_include_directories(cubp
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cubp
    PUBLIC
)

set_target_properties(cubp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
