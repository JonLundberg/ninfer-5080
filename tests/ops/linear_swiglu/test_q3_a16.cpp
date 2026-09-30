#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // Covers the T=1 GEMV, every small-T MMA width (8- and 16-column tiles), and
        // 32-token-tile prefill with each tail route (small-T MMA, 16 + GEMV, 16 + MMA).
        constexpr std::array<std::int32_t, 20> kTokenCases{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 37, 48, 49, 50,
        };

        const int failures = run_profile(
            "LinearSwiGLU Q3_A16",
            {
                QType::Q3G64_F16S,
                34816,
                5120,
                17408,
                1403U,
                ActivationCompute::A16
            },
            kTokenCases);

        std::cout
            << (failures == 0 ? "OK" : "FAIL")
            << " LinearSwiGLU Q3_A16 correctness\n";

        return failures == 0 ? 0 : 1;

    } catch (const std::exception& error) {

        std::cerr
            << "LinearSwiGLU Q3_A16 test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}
