if not is_plat("android") then


target("cuda-pipe")

    set_kind("binary")
    add_files("main.cu")

    add_deps("builtin-apps")

    add_packages("spdlog")


    -- add_packages("concurrentqueue")
    -- add_packages("cli11")
    add_packages("glm")

    add_cugencodes("native")

    add_packages("openmp")
    

target_end()

end