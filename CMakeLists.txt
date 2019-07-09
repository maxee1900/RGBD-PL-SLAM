cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)


SET(CMAKE_BUILD_TYPE Release)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
add_definitions(-DCOMPILEDWITHC11)
if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -g -march=native -Wno-deprecated-declarations")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -O3 -march=native -Wno-deprecated-declarations")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#find_package(OpenCV 3.0 REQUIRED)
find_package(OpenCV 3.3 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 2.4.3 QUIET)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
  endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
#add
find_package(PCL 1.7 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}

${PCL_INCLUDE_DIRS}
#以下这一行是line描述子的头文件 不知道要不要加
#${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/include
)
#@ 这里没有和Thirdparty中的line_descriptor中的文件构建关系 也就是说没用到？

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cpp
src/Tracking.cpp
src/LocalMapping.cpp
src/LoopClosing.cpp
src/ORBextractor.cpp
src/ORBmatcher.cpp
src/FrameDrawer.cpp
src/Converter.cpp
src/MapPoint.cpp
src/KeyFrame.cpp
src/Map.cpp
src/MapDrawer.cpp
src/Optimizer.cpp
src/PnPsolver.cpp
src/Frame.cpp
src/KeyFrameDatabase.cpp
src/Sim3Solver.cpp
src/Viewer.cpp

#add 库中增加了三个源文件
src/ExtractLineSegment.cpp
src/MapLine.cpp
src/LSDmatcher.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so

#  ${PROJECT_SOURCE_DIR/Thirdparty/line_descriptor/lib/liblinedesc.so}     #如果用第三方库中的直线描述子 改动两个地方：一个是包含头文件一个是连接库文件
)



add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})



#老子能在两周内搞定程序 然后将自己的算法加进去
#deadline：7.14
#任务：完成算法实验，验证算法
#一共二十天的时间