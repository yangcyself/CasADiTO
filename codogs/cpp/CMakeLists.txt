project(cppGen)
cmake_minimum_required(VERSION 3.7)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC") # must set the release flag, or it will add -O3 by default

link_directories(/usr/local/lib)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})

add_subdirectory(generated)    # nlpFltJmp

# target_compile_options(nlp PRIVATE "-pg")
# target_link_options(nlp PRIVATE "-pg")

add_executable(nlp main.cpp tonlp.cpp )
target_link_libraries(nlp hrl)
target_link_libraries(nlp ipopt)
target_link_libraries (nlp Eigen3::Eigen)
target_include_directories(nlp PRIVATE /usr/local/include/coin-or /usr/local/include/coin-or/mumps)


# add_library(hrlDyn SHARED hrlDyn.cpp tonlp.cpp )
# target_link_libraries(hrlDyn ipopt)
# target_link_libraries (hrlDyn Eigen3::Eigen)
# target_link_libraries(hrlDyn 
#                     "-Wl,--whole-archive"
#                     hrl
#                     "-Wl,--no-whole-archive")
# target_include_directories(hrlDyn PRIVATE /usr/local/include/coin-or /usr/local/include/coin-or/mumps)


add_library(hrlDyn hrlDyn.cpp tonlp.cpp )
target_link_libraries(hrlDyn ipopt)
target_link_libraries (hrlDyn Eigen3::Eigen)
target_link_libraries(hrlDyn hrl)
target_include_directories(hrlDyn PRIVATE /usr/local/include/coin-or /usr/local/include/coin-or/mumps)


add_executable(testhrl testHrl.cpp)
target_link_libraries(testhrl hrlDynplug)


add_library(hrlDynplug SHARED hrlDynplugin.cpp)
target_link_libraries(hrlDynplug hrlDyn)


add_executable(testhrl_plug testHrl_plug.cpp)
target_link_libraries(testhrl_plug hrlDynplug)


# target_link_libraries (testhrl Eigen3::Eigen)
# target_link_libraries(testhrl ipopt)
# target_include_directories(testhrl PRIVATE /usr/local/include/coin-or /usr/local/include/coin-or/mumps)
