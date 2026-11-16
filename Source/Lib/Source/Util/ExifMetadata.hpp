// Copyright (c) 2026 Minmin Gong
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace AIHoloImager
{
    // Exif metadata is a TIFF structure. It's stored in an APP1 segment of a JPEG, or in an eXIf chunk of a PNG.
    std::optional<std::vector<std::byte>> LoadExifMetadata(const std::filesystem::path& path);
    bool SaveExifMetadata(const std::filesystem::path& path, std::span<const std::byte> exif);
} // namespace AIHoloImager
