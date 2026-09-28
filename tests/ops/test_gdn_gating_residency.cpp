#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>

using namespace ninfer::ops::detail;

namespace {

int check(const std::string& label, bool actual, bool expected) {
    if (actual == expected) { return 0; }
    std::cout << "FAIL " << label << ": got " << (actual ? "resident" : "not resident")
              << ", expected " << (expected ? "resident" : "not resident") << "\n";
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // 27B geometry boundary from the dual-GPU bug. T=1537 -> ceil(1537/128)=13 column tiles *
    // 3 row tiles * split2 = 78 CTAs. A 36-SM device budgets 36*2=72 resident CTAs (reject);
    // a 40-SM device budgets 40*2=80 (admit). This is the exact mismatch that produced
    // cudaErrorCooperativeLaunchTooLarge when the SM count was read from the wrong device.
    const Bf16GdnGatingScheduleId split2 = Bf16GdnGatingScheduleId::MmaCooperativeSplit2;
    failures += check("27b T=1537 36SM rejects", bf16_gdn_gating_27_resident(split2, 1537, 36), false);
    failures += check("27b T=1537 40SM admits", bf16_gdn_gating_27_resident(split2, 1537, 40), true);
    failures += check("27b T=1537 70SM admits", bf16_gdn_gating_27_resident(split2, 1537, 70), true);

    // A zero SM count (device discovery failure) must reject every cooperative schedule.
    failures += check("27b zero-SM rejects", bf16_gdn_gating_27_resident(split2, 1537, 0), false);
    failures += check(
        "35b zero-SM rejects",
        bf16_gdn_gating_35_resident(Bf16GdnGatingScheduleId::MmaCooperativeSplit32, 1537, 0), false);

    // 35B geometry: Split32 admits 2 CTAs/SM, other splits 4 CTAs/SM. T=1537 ->
    // ceil(1537/64)=24 column tiles * 2 row tiles * split32 = 1536 CTAs, far above a 36-SM
    // budget of 72, so it is rejected on the small device.
    failures += check(
        "35b split32 T=1537 36SM rejects",
        bf16_gdn_gating_35_resident(Bf16GdnGatingScheduleId::MmaCooperativeSplit32, 1537, 36), false);

    // A zero-SM device never admits the 35B cooperative Split32 residency (deterministic).
    failures += check("35b zero-SM Split32 not resident",
                      bf16_gdn_gating_35_resident(Bf16GdnGatingScheduleId::MmaCooperativeSplit32, 8, 0),
                      false);

    // Fused 35B norm-gating with cols <= 16 must fall back to Composed (not throw) when the
    // cooperative Split32 candidate is unavailable. On a host with no CUDA device,
    // device_sm_count() is 0, which is exactly that condition; the already-resolved Composed
    // control plan is retained instead of resolving the non-resident Split32 candidate.
    {
        int device_count = 0;
        const bool no_cuda =
            cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0;
        if (no_cuda) {
            const Bf16GdnGatingProblem small35{32, 2048, 8}; // is_35, cols=8 <= 16
            bool threw = false;
            Bf16GdnNormGatingScheduleId sched = Bf16GdnNormGatingScheduleId::Composed;
            try {
                sched = bf16_gdn_norm_gating_resolve_plan(small35).schedule;
            } catch (...) {
                threw = true;
            }
            failures += check("35b norm-gating zero-SM does not throw", !threw, true);
            failures += check("35b norm-gating zero-SM -> Composed fallback",
                              sched == Bf16GdnNormGatingScheduleId::Composed, true);
        }
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating_residency\n";
    return failures == 0 ? 0 : 1;
}
