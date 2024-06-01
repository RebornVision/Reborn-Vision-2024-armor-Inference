cmake_minimum_required(VERSION 3.16)
project(INFERENCE)

set(CMAKE_CXX_STANDARD 20)
set(ngraph_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
set(InferenceEngine_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake")

set(INC_DIR "/opt/intel/openvino_2023.3.0/runtime/include")
set(LINK_DIR "/opt/intel/openvino_2023.3.0/runtime/lib/intel64")



find_package(OpenCV 4 REQUIRED)
#find_package(InferenceEngine REQUIRED)
#find_package(OpenVINO)
#find_package(OpenVINO REQUIRED)


include_directories(
        #OpenVINO推理引擎的头文件

        "/opt/intel/openvino_2023.3.0/runtime/include/ie/"
        "/opt/intel/openvino_2023/runtime/include/ngraph/"
        "/opt/intel/openvino_2023.3.0/runtime/include/openvino/"


)

include_directories(
        ${OpenCV_INCLUDE_DIRS}

)

include_directories(
        ${INC_DIR}

)
link_directories(
        ${LINK_DIR}
)
include_directories(
        ${PROJECT_SOURCE_DIR}/include/
)
link_libraries(libopenvino.so)

add_executable(INFERENCE
        src/video_detect_test.cpp
        src/tradition.cpp
        src/inference.cpp
        src/number_classifier.cpp

)
target_link_libraries(

        INFERENCE

        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}


)



