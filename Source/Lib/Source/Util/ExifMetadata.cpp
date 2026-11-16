// Copyright (c) 2026 Minmin Gong
//

#include "ExifMetadata.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>

namespace AIHoloImager
{
    namespace
    {
        constexpr std::array<std::byte, 2> JpegSoi = {std::byte{0xFF}, std::byte{0xD8}};
        constexpr std::array<std::byte, 8> PngSignature = {std::byte{0x89}, std::byte{'P'}, std::byte{'N'}, std::byte{'G'}, std::byte{0x0D},
            std::byte{0x0A}, std::byte{0x1A}, std::byte{0x0A}};
        constexpr std::array<std::byte, 6> ExifHeader = {
            std::byte{'E'}, std::byte{'x'}, std::byte{'i'}, std::byte{'f'}, std::byte{0}, std::byte{0}};
        constexpr std::array<std::byte, 4> PngIhdrChunkType = {std::byte{'I'}, std::byte{'H'}, std::byte{'D'}, std::byte{'R'}};
        constexpr std::array<std::byte, 4> PngExifChunkType = {std::byte{'e'}, std::byte{'X'}, std::byte{'I'}, std::byte{'f'}};

        constexpr size_t MaxJpegSegmentSize = 0xFFFF;

        using ByteVec = std::vector<std::byte>;

        bool ReadWholeFile(const std::filesystem::path& path, ByteVec& data)
        {
            std::ifstream file(path, std::ios_base::binary);
            if (!file)
            {
                return false;
            }

            file.seekg(0, std::ios_base::end);
            const std::streamoff size = file.tellg();
            if (size < 0)
            {
                return false;
            }

            file.seekg(0, std::ios_base::beg);
            data.resize(static_cast<size_t>(size));
            file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));

            return static_cast<bool>(file);
        }

        bool WriteWholeFile(const std::filesystem::path& path, std::span<const std::byte> data)
        {
            std::ofstream file(path, std::ios_base::binary | std::ios_base::trunc);
            if (!file)
            {
                return false;
            }

            file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));

            return static_cast<bool>(file);
        }

        bool BeginsWith(std::span<const std::byte> data, std::span<const std::byte> prefix)
        {
            return (data.size() >= prefix.size()) && std::equal(prefix.begin(), prefix.end(), data.begin());
        }

        void AppendBeUint32(ByteVec& data, uint32_t value)
        {
            data.push_back(static_cast<std::byte>((value >> 24) & 0xFFU));
            data.push_back(static_cast<std::byte>((value >> 16) & 0xFFU));
            data.push_back(static_cast<std::byte>((value >> 8) & 0xFFU));
            data.push_back(static_cast<std::byte>(value & 0xFFU));
        }

        uint32_t ReadBeUint32(std::span<const std::byte> data, size_t offset)
        {
            return (static_cast<uint32_t>(std::to_integer<uint8_t>(data[offset])) << 24) |
                   (static_cast<uint32_t>(std::to_integer<uint8_t>(data[offset + 1])) << 16) |
                   (static_cast<uint32_t>(std::to_integer<uint8_t>(data[offset + 2])) << 8) |
                   static_cast<uint32_t>(std::to_integer<uint8_t>(data[offset + 3]));
        }

        struct JpegSegment
        {
            uint8_t marker = 0;
            size_t begin = 0;
            size_t end = 0;
        };

        // Collects the segments in front of the entropy coded data.
        std::vector<JpegSegment> StatJpegHeaderSegments(std::span<const std::byte> data)
        {
            std::vector<JpegSegment> segments;

            size_t offset = JpegSoi.size();
            while ((offset + 4) <= data.size())
            {
                if (data[offset] != std::byte{0xFF})
                {
                    break;
                }

                const uint8_t marker = std::to_integer<uint8_t>(data[offset + 1]);
                if (marker == 0xFF)
                {
                    // Fill byte
                    ++offset;
                    continue;
                }
                if ((marker == 0x01) || ((marker >= 0xD0) && (marker <= 0xD9)))
                {
                    // Markers without a payload
                    offset += 2;
                    continue;
                }
                if (marker == 0xDA)
                {
                    // Start of scan, the rest of the file is entropy coded data
                    break;
                }

                const size_t size = (static_cast<size_t>(std::to_integer<uint8_t>(data[offset + 2])) << 8) |
                                    static_cast<size_t>(std::to_integer<uint8_t>(data[offset + 3]));
                if (size < 2)
                {
                    break;
                }

                segments.push_back({marker, offset, offset + 2 + size});
                offset += 2 + size;
            }

            return segments;
        }

        bool IsJpegExifSegment(std::span<const std::byte> data, const JpegSegment& segment)
        {
            return (segment.marker == 0xE1) && BeginsWith(data.subspan(segment.begin + 4), ExifHeader);
        }

        std::optional<ByteVec> LoadJpegExif(std::span<const std::byte> data)
        {
            for (const JpegSegment& segment : StatJpegHeaderSegments(data))
            {
                if (IsJpegExifSegment(data, segment))
                {
                    const size_t exif_begin = segment.begin + 4 + ExifHeader.size();
                    return ByteVec(data.begin() + exif_begin, data.begin() + segment.end);
                }
            }

            return std::nullopt;
        }

        struct PngChunk
        {
            std::array<std::byte, 4> type = {};
            size_t begin = 0;
            size_t end = 0;
            size_t data_begin = 0;
        };

        std::vector<PngChunk> StatPngChunks(std::span<const std::byte> data)
        {
            std::vector<PngChunk> chunks;

            size_t offset = PngSignature.size();
            while ((offset + 12) <= data.size())
            {
                const size_t size = ReadBeUint32(data, offset);
                if ((offset + 12 + size) > data.size())
                {
                    break;
                }

                PngChunk& chunk = chunks.emplace_back();
                std::copy(data.begin() + offset + 4, data.begin() + offset + 8, chunk.type.begin());
                chunk.begin = offset;
                chunk.end = offset + 12 + size;
                chunk.data_begin = offset + 8;

                offset += 12 + size;
            }

            return chunks;
        }

        std::optional<ByteVec> LoadPngExif(std::span<const std::byte> data)
        {
            for (const PngChunk& chunk : StatPngChunks(data))
            {
                if (chunk.type == PngExifChunkType)
                {
                    return ByteVec(data.begin() + chunk.data_begin, data.begin() + chunk.end - 4);
                }
            }

            return std::nullopt;
        }

        uint32_t CalcCrc32(std::span<const std::byte> data)
        {
            static const std::array<uint32_t, 256> table = [] {
                std::array<uint32_t, 256> ret{};
                for (uint32_t i = 0; i < ret.size(); ++i)
                {
                    uint32_t crc = i;
                    for (uint32_t bit = 0; bit < 8; ++bit)
                    {
                        crc = ((crc & 1U) != 0) ? (0xEDB88320U ^ (crc >> 1)) : (crc >> 1);
                    }
                    ret[i] = crc;
                }
                return ret;
            }();

            uint32_t crc = 0xFFFFFFFFU;
            for (const std::byte b : data)
            {
                crc = table[(crc ^ std::to_integer<uint8_t>(b)) & 0xFFU] ^ (crc >> 8);
            }

            return crc ^ 0xFFFFFFFFU;
        }

        bool SaveJpegExif(const std::filesystem::path& path, std::span<const std::byte> data, std::span<const std::byte> exif)
        {
            const size_t segment_size = 2 + ExifHeader.size() + exif.size();
            if (segment_size > MaxJpegSegmentSize)
            {
                return false;
            }

            ByteVec output;
            output.reserve(data.size() + segment_size);
            output.insert(output.end(), JpegSoi.begin(), JpegSoi.end());

            output.push_back(std::byte{0xFF});
            output.push_back(std::byte{0xE1});
            output.push_back(static_cast<std::byte>((segment_size >> 8) & 0xFFU));
            output.push_back(static_cast<std::byte>(segment_size & 0xFFU));
            output.insert(output.end(), ExifHeader.begin(), ExifHeader.end());
            output.insert(output.end(), exif.begin(), exif.end());

            const std::vector<JpegSegment> segments = StatJpegHeaderSegments(data);
            for (const JpegSegment& segment : segments)
            {
                if (!IsJpegExifSegment(data, segment))
                {
                    output.insert(output.end(), data.begin() + segment.begin, data.begin() + segment.end);
                }
            }

            output.insert(output.end(), data.begin() + (segments.empty() ? JpegSoi.size() : segments.back().end), data.end());

            return WriteWholeFile(path, output);
        }

        bool SavePngExif(const std::filesystem::path& path, std::span<const std::byte> data, std::span<const std::byte> exif)
        {
            ByteVec output;
            output.reserve(data.size() + exif.size() + 12);
            output.insert(output.end(), PngSignature.begin(), PngSignature.end());

            ByteVec chunk;
            chunk.reserve(exif.size() + 12);
            AppendBeUint32(chunk, static_cast<uint32_t>(exif.size()));
            chunk.insert(chunk.end(), PngExifChunkType.begin(), PngExifChunkType.end());
            chunk.insert(chunk.end(), exif.begin(), exif.end());
            AppendBeUint32(chunk, CalcCrc32(std::span(chunk).subspan(4)));

            bool has_ihdr = false;
            for (const PngChunk& old_chunk : StatPngChunks(data))
            {
                if (old_chunk.type == PngExifChunkType)
                {
                    continue;
                }

                output.insert(output.end(), data.begin() + old_chunk.begin, data.begin() + old_chunk.end);

                if (!has_ihdr && (old_chunk.type == PngIhdrChunkType))
                {
                    // eXIf must follow IHDR
                    output.insert(output.end(), chunk.begin(), chunk.end());
                    has_ihdr = true;
                }
            }

            if (!has_ihdr)
            {
                return false;
            }

            return WriteWholeFile(path, output);
        }
    } // namespace

    std::optional<std::vector<std::byte>> LoadExifMetadata(const std::filesystem::path& path)
    {
        ByteVec data;
        if (!ReadWholeFile(path, data))
        {
            return std::nullopt;
        }

        if (BeginsWith(data, JpegSoi))
        {
            return LoadJpegExif(data);
        }
        if (BeginsWith(data, PngSignature))
        {
            return LoadPngExif(data);
        }

        return std::nullopt;
    }

    bool SaveExifMetadata(const std::filesystem::path& path, std::span<const std::byte> exif)
    {
        ByteVec data;
        if (!ReadWholeFile(path, data))
        {
            return false;
        }

        if (BeginsWith(data, JpegSoi))
        {
            return SaveJpegExif(path, data, exif);
        }
        if (BeginsWith(data, PngSignature))
        {
            return SavePngExif(path, data, exif);
        }

        return false;
    }
} // namespace AIHoloImager
