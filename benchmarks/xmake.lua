-- Common benchmark configuration
rule("benchmark_config")
    on_load(function (target)
        target:set("kind", "binary")
        target:set("group", "benchmarks")
        target:add("includedirs", "$(projectdir)/builtin-apps/")
        target:add("includedirs", "$(projectdir)")
        target:add("packages", "benchmark")
        target:add("packages", "glm")
        target:add("packages", "spdlog")
    end)
rule_end()


includes("bm-tree")
includes("bm-cifar-sparse")
includes("bm-cifar-dense")