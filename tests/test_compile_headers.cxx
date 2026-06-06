// Compile-only test: verifies all modified headers parse without errors.
// If this file compiles, the header changes are syntactically valid.
//
// Why this test exists: the sliceiter branch adds new headers (parse_schedule.h)
// and modifies existing ones (register_dwi_to_b0.h, rigid_register_images.h).
// A missing #include or forward declaration in these headers breaks compilation
// of TORTOISEProcess and all tools that transitively include them. This test
// catches those errors without building the full pipeline.

#include "register_dwi_to_b0.h"
#include "rigid_register_images.h"
#include "parse_schedule.h"

// register_dwi_to_slice.h and register_dwi_to_slice_cuda.h are header-only
// implementations (contain function bodies). Including them here would cause
// linker issues without their full dependency chain. They are verified
// transitively when TORTOISEProcess builds.

int main()
{
    // Verify parse_schedule.h compiles and basic types work
    std::vector<float> fs = parse_float_schedule("1.0", {0.0f});
    std::vector<int> is = parse_int_schedule("2", {0});
    float fv = schedule_value(fs, 0);
    int iv = schedule_value(is, 0);
    (void)fv;
    (void)iv;

    return 0;
}
