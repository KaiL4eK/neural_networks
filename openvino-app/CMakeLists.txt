cmake_minimum_required(VERSION 3.5)

project(openvino-app)

find_package(OpenCV 4.0.0 REQUIRED)
find_package(InferenceEngine 2.0 REQUIRED)
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
)

target_link_libraries(${YOLO_LIB} 
PUBLIC     
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES}
)

# Create main app

add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/yolo_ov.cpp
)

target_link_libraries(${PROJECT_NAME} 
    ${InferenceEngine_LIBRARIES} 
    IE::ie_cpu_extension
    ${YOLO_LIB} 
)

target_include_directories(${PROJECT_NAME} 
PRIVATE
    include
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
)

