target("pipe-cifar-sparse-vk")
    set_kind("binary")
    add_files("main.cpp")

    add_includedirs("$(projectdir)/builtin-apps")
    add_includedirs("$(projectdir)")

    add_deps("builtin-apps")
    add_deps("builtin-apps-vulkan")

    add_packages("spdlog")
    add_packages("vulkan-hpp", "vulkan-memory-allocator")

    add_packages("concurrentqueue")
    -- add_packages("cli11")

    if is_plat("android") then
      on_run(run_on_android)
    end