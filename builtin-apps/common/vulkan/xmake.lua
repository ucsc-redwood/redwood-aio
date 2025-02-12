-- add_requires("vulkan-hpp 1.3.290")
add_requires("vulkan-hpp", "vulkan-memory-allocator")

target("kiss-vk")
    set_kind("static")
    
    add_headerfiles({
        "algorithm.hpp",
        "base_engine.hpp",
        "engine.hpp",
        "sequence.hpp",
        "vk.hpp",
        "vma_pmr.hpp",
    })

    add_files({
        "algorithm.cpp",
        "base_engine.cpp",
        "sequence.cpp",
        "vma_pmr.cpp",
    })

    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("spdlog")

target_end()


