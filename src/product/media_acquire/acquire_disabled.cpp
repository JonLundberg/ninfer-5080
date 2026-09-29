// Built when NINFER_ENABLE_MEDIA=OFF: media acquisition is unavailable (no libcurl).
#include "product/media_acquire/acquire.h"

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw std::runtime_error("media acquisition is disabled in this build (NINFER_ENABLE_MEDIA=OFF)");
}

} // namespace ninfer::product::media_acquire
