// Built when NINFER_ENABLE_MEDIA=OFF: media decode is unavailable (no FFmpeg).
#include "media/decode/decode.h"

namespace ninfer::media::decode {

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::runtime_error("media decode is disabled in this build (NINFER_ENABLE_MEDIA=OFF)");
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::runtime_error("media decode is disabled in this build (NINFER_ENABLE_MEDIA=OFF)");
}

} // namespace ninfer::media::decode
