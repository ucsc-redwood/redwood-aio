if not is_plat("android") then


target("test-cuda-tree")

    set_kind("binary")
    add_files({
        "main.cu",
        "../../builtin-apps/common/cuda/cu_mem_resource.cu",
    })

    add_includedirs("$(projectdir)/builtin-apps/")
    add_deps("builtin-apps")
    add_deps("builtin-apps-cuda")

    add_packages("spdlog")

    -- Add CUDA flags
    add_cuflags("-Xcompiler", "-fopenmp", {force = true})  -- Enable OpenMP for the CUDA compiler
    add_ldflags("-fopenmp", {force = true})  -- Link against OpenMP library
    add_packages("openmp")

    add_packages("concurrentqueue")
    add_packages("cli11")
    add_packages("glm")

    add_cugencodes("native")
    add_packages("openmp")
    

target_end()

end