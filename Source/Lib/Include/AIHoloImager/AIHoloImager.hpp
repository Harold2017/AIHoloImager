// Copyright (c) 2024 Minmin Gong
//

#pragma once

#include <filesystem>
#include <memory>

#include "AIHoloImager/Mesh.hpp"

namespace AIHoloImager
{
    class AIHoloImager
    {
    public:
        enum class DeviceType
        {
            Cpu,
            Cuda,
        };
        enum class Api
        {
            D3D12,
            Vulkan,

            Auto,
        };
        enum class FeatureMatcher
        {
            SuperPointLightGlue,
            LoMa,
        };

    public:
        AIHoloImager(DeviceType device, Api api, const std::filesystem::path& tmp_dir, bool gpu_debug = false);
        AIHoloImager(const AIHoloImager& rhs) = delete;
        AIHoloImager(AIHoloImager&& rhs) noexcept;
        ~AIHoloImager() noexcept;

        AIHoloImager& operator=(const AIHoloImager& rhs) = delete;
        AIHoloImager& operator=(AIHoloImager&& rhs) noexcept;

        Mesh Generate(const std::filesystem::path& input_path, uint32_t texture_size, bool no_delight = false,
            FeatureMatcher matcher = FeatureMatcher::SuperPointLightGlue);

        // Delights every image in input_path (a directory or a single image file) and saves the results to output_dir,
        // keeping the Exif metadata of the original images. Images larger than max_size are delighted at a reduced
        // resolution and scaled back, to bound the memory usage. 0 means no limit.
        void Delight(const std::filesystem::path& input_path, const std::filesystem::path& output_dir, uint32_t max_size = 2048);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace AIHoloImager
