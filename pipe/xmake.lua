-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Pipeline
-- ----------------------------------------------------------------

add_requires("concurrentqueue")

rule("pipe_config")
    on_load(function (target)
        target:set("kind", "binary")
        target:set("group", "pipe")

        target:add("includedirs", "$(projectdir)")
        
        target:add("packages", "concurrentqueue")
    end)
rule_end()

-- ----------------------------------------------------------------
-- Pipeline Targets
-- ----------------------------------------------------------------

includes("cifar-dense-vk")
includes("cifar-sparse-vk")