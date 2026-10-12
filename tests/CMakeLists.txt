add_executable(cldiff_tests
    test_implied_vol.cpp
    test_cos_pricer.cpp
    test_heston_objective.cpp
    test_estimator.cpp
    test_gbm_vs_bs.cpp
    test_akdl.cpp
)
target_link_libraries(cldiff_tests PRIVATE cldiff Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(cldiff_tests)
