cmake_minimum_required(VERSION 3.16)
project(rm_auto_aim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 源码是 UTF-8，中文 Windows 下 MSVC 默认按 GBK(936) 读，
# 会把中文字符的尾字节误当成代码，产生上百条假语法错误。这条必须保留。
if(MSVC)
  add_compile_options(/utf-8)
  # 让 <cmath> 暴露 M_PI（MSVC 默认不定义）
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

# 用到的 OpenCV 组件：
#   core      矩阵与基础类型
#   imgproc   绘制、颜色转换
#   imgcodecs 读写图片
#   videoio   读写视频（VideoCapture / VideoWriter）
#   highgui   预留（调试窗口）
#   calib3d   solvePnP / projectPoints / Rodrigues
#   dnn       ONNX 推理
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui calib3d dnn)
message(STATUS "OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR}")

file(GLOB RM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(auto_aim ${RM_SOURCES})

target_include_directories(auto_aim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(auto_aim PRIVATE ${OpenCV_LIBS})

# 把工程根目录写进程序。
# 运行时所有相对路径都由 config.cpp 的 resolvePath() 拼成
#   RM_PROJECT_ROOT + "/" + 配置里的路径
# 因此配置里的路径必须相对「工程根目录」，不能带 ../
target_compile_definitions(auto_aim PRIVATE
    RM_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# 构建后把 config/ 拷到 exe 旁边，方便单独拷贝程序运行
add_custom_command(TARGET auto_aim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/config"
            "$<TARGET_FILE_DIR:auto_aim>/config")

if(WIN32)
  # 直接链接到的 DLL（opencv_world4100.dll 等）
  add_custom_command(TARGET auto_aim POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:auto_aim>"
              "$<TARGET_RUNTIME_DLLS:auto_aim>"
      COMMAND_EXPAND_LISTS)

  # videoio 的后端（FFmpeg / MSMF）是运行时按需加载的，不在链接依赖里，
  # 必须单独拷；缺 FFmpeg 时表现为能读 avi、但写 mp4 失败。
  if(DEFINED OpenCV_LIB_PATH)
    file(GLOB _ocv_videoio_dlls "${OpenCV_LIB_PATH}/../bin/opencv_videoio_*_64.dll")
    list(FILTER _ocv_videoio_dlls EXCLUDE REGEX "d\\.dll$")
    if(_ocv_videoio_dlls)
      add_custom_command(TARGET auto_aim POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:auto_aim>"
                  ${_ocv_videoio_dlls}
          COMMAND_EXPAND_LISTS)
    endif()
  endif()
endif()
