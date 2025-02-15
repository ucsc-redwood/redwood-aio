target("pipe-cifar-dense-vk")
    set_kind("binary")

    add_files({
        "main.cpp",
        "generated-code/*.cpp",
    })

    add_includedirs("$(projectdir)")

    add_deps("builtin-apps")
    add_deps("builtin-apps-vulkan")

    add_packages("spdlog")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")

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