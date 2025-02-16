target("try-cifar-dense-vk")
    set_kind("binary")
    add_files("main.cpp")
    
    add_includedirs("$(projectdir)")

    add_deps("builtin-apps")
    add_deps("builtin-apps-vulkan")
    
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("spdlog")

    add_rules("run_on_android")
