#pragma once

#include "builtin-apps/conf.hpp"

template <int Stage>
concept ValidStage = (Stage >= 1) && (Stage <= 9);

template <int Start, int End>
concept ValidStageRange = ValidStage<Start> && ValidStage<End> && (Start <= End);

template <ProcessorType processor_type>
concept ValidProcessorType =
    (processor_type == ProcessorType::kLittleCore) ||
    (processor_type == ProcessorType::kMediumCore) || (processor_type == ProcessorType::kBigCore);
