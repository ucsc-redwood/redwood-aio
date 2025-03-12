target("omp-sort")
    add_rules("common_flags", "run_on_android")
    add_includedirs("$(projectdir)/")
    add_files("main.cpp")
