
add_requires("benchmark")

target("bm")
    set_kind("binary")
    add_files("*.cpp")
    add_packages("builtin-apps")
    add_packages("benchmark")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    if is_plat("android") then
      on_run(run_on_android)
    end

