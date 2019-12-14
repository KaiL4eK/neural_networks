cmake_minimum_required(VERSION 3.5)

project(inference_samples)

find_package(OpenCV 3.3.0 REQUIRED)
find_package(Boost 1.6.5 REQUIRED COMPONENTS program_options filesystem)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library

set(YOLO_LIB yolo_cpp)
add_library(${YOLO_LIB} SHARED
            src/yolo.cpp
)
target_include_directories(${YOLO_LIB} 
PUBLIC 
    include
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
)
target_link_libraries(${YOLO_LIB} 
PUBLIC     
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES}
)

# Create main app OpenVINO

find_package(InferenceEngine 2.0 REQUIRED)

set(OPENVINO_APP app)
add_executable(${OPENVINO_APP} 
    src/main.cpp
    src/yolo_ov.cpp
)
target_link_libraries(${OPENVINO_APP} 
    ${InferenceEngine_LIBRARIES} 
    IE::ie_cpu_extension
    ${YOLO_LIB}
)
target_include_directories(${OPENVINO_APP} 
PRIVATE
    include
)
