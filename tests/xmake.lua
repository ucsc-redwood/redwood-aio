add_requires("gtest")

target("tests")
    set_kind("binary")
    set_group("test")
    add_files("omp_sort/main.cpp")
    add_packages("gtest")
