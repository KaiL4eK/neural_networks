cmake_minimum_required(VERSION 3.5)

project(openvino-app)

find_package(OpenCV 4.0.0 REQUIRED)
find_package(InferenceEngine 2.0 REQUIRED)
find_package(Boost 1.6.5 REQUIRED COMPONENTS program_options filesystem)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} 
    main.cpp
    yolo.cpp
)

target_link_libraries(${PROJECT_NAME} 
    ${InferenceEngine_LIBRARIES} 
    IE::ie_cpu_extension 
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
)

