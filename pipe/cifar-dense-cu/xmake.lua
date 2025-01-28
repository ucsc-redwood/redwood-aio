if not is_plat("android") then


target("pipe-cifar-dense-cu")

    set_kind("binary")
    add_files({
        "main.cu",
        "../../builtin-apps/common/cuda/cu_mem_resource.cu",
    })

    add_deps("builtin-apps")
    add_deps("builtin-apps-cuda")
    add_includedirs("../../builtin-apps")

    add_packages("spdlog")


    add_packages("concurrentqueue")
    add_packages("cli11")

    add_cugencodes("native")

    add_packages("openmp")
    

target_end()

end