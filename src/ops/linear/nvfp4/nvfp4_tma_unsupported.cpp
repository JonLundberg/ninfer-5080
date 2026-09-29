// MSVC build: the NVFP4 TMA kernels pass 128-byte-aligned CUtensorMap descriptors by value,
// which the MSVC x64 host launch stubs cannot express (C2719). NVFP4 artifacts are unsupported.
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
[[noreturn]] void unsupported() {
    throw std::runtime_error("NVFP4 TMA kernels are not available in MSVC builds");
}
} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) { unsupported(); }
void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, std::int32_t, float,
                                     cudaStream_t) { unsupported(); }
void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) { unsupported(); }
void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) { unsupported(); }
void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                         std::int32_t, float, cudaStream_t) { unsupported(); }

} // namespace ninfer::ops::detail
