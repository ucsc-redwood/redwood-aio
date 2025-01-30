target("threads")
    set_kind("binary")
    add_files("main.cpp")

    add_deps("builtin-apps")

    add_packages("spdlog")


    add_packages("concurrentqueue")
    add_packages("cli11")
    add_packages("glm")

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
target_end()

