cmake_minimum_required(VERSION 3.10)
project(LLMInferDemo CXX)

set(CMAKE_CXX_STANDARD 11)

# 指定头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 收集所有 cpp 文件
file(GLOB SRC_FILES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

add_executable(llm_infer_template
    main.cpp
    src/llm_infer_pipeline.cpp
    ${SRC_FILES}
)

target_include_directories(llm_infer_template PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
