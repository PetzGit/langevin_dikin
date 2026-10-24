cmake_minimum_required(VERSION 3.14)
project(cldiff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies, fetched at configure time ---
include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
# Eigen ships its own CMakeLists.txt that builds tests/docs we don't want;
# these two lines skip that and just make the headers available.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)
# Treat Eigen as a system header so its internal warnings (e.g. GCC 13's
# false-positive -Wmaybe-uninitialized in the triangular kernels) don't trip
# -Werror on our own targets.
get_target_property(EIGEN_INCLUDES eigen INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(eigen PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${EIGEN_INCLUDES}")

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(catch2)

# Strict warnings on every target defined below (library, tests,
# experiments). Set after the FetchContent calls so Eigen and Catch2's own
# targets don't inherit it, and Eigen is a system include so its headers are
# exempt too.
add_compile_options(-Wall -Wextra -Werror)

# --- Library ---
add_library(cldiff
    # core
    src/core/rng.cpp
    src/core/estimator.cpp

    # optim -- generic, no finance
    src/optim/objective.cpp
    src/optim/polyhedron.cpp
    src/optim/log_space_objective.cpp
    src/optim/tempered_objective.cpp
    src/optim/geometry/kinetic_dikin.cpp
    src/optim/schedules/three_phase_log.cpp
    src/optim/schedules/parallel_floor.cpp
    src/optim/optimisers/annealed_langevin.cpp
    src/optim/optimisers/projected_gradient_descent.cpp
    src/optim/optimisers/annealed_akdl.cpp
    src/optim/samplers/dikin_langevin.cpp
    src/optim/samplers/akdl.cpp

    # market
    src/market/market_data.cpp
    src/market/implied_vol.cpp

    # models
    src/models/heston/characteristic_function.cpp
    src/models/heston/cumulants.cpp
    src/models/heston/feller.cpp
    src/models/heston/heston_cos_pricer.cpp
    src/models/heston/full_truncation_scheme.cpp
    src/models/gbm/gbm_exact.cpp
    src/models/ou/ou_exact.cpp

    # products
    src/products/european_call.cpp

    # pricing
    src/pricing/fourier/cos_pricer.cpp
    #src/pricing/pde/grid.cpp
    src/pricing/monte_carlo/mc_harness.cpp

    # calibration -- where pricing meets optim
    src/calibration/calibration_objective.cpp
    src/calibration/implied_vol_squared_loss.cpp
    src/calibration/feller_penalty.cpp
    src/calibration/heston_objective.cpp
    src/calibration/corrected_log_space_heston_objective.cpp
)
target_include_directories(cldiff PUBLIC include)
target_link_libraries(cldiff PUBLIC Eigen3::Eigen)

# --- Tests ---
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
enable_testing()
add_subdirectory(tests)

add_subdirectory(experiments)
