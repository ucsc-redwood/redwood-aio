-- set_policy("build.cuda.devlink", true)

-- target("cifar_dense_cuda")
--     set_kind("binary")
--     add_files("main.cu")
--     add_deps("builtin-apps")
--     add_deps("builtin-apps-cuda")

--     add_cugencodes("native")

--     add_packages("spdlog")
-- target_end()
