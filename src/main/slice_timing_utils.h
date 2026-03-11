#ifndef _SLICE_TIMING_UTILS_H
#define _SLICE_TIMING_UTILS_H

#include <vnl/vnl_matrix.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../external_src/json_nlohmann/json.hpp"

using json = nlohmann::json;

// Convert SliceTiming array from JSON to FSL-style slspec matrix.
// Extracted from DIFFPREP::ParseJSONForSliceTiming — pure function with no
// DIFFPREP dependencies (takes json, returns vnl_matrix<int>).
inline vnl_matrix<int> ParseJSONForSliceTiming(json cjson)
{
    std::vector<float> slice_timings = cjson["SliceTiming"];
    std::vector<float> slice_timings_orig = slice_timings;

    std::sort(slice_timings.begin(), slice_timings.end());
    auto last = std::unique(slice_timings.begin(), slice_timings.end());
    slice_timings.erase(last, slice_timings.end());

    int N_unique_times = slice_timings.size();
    int MB = 0;
    for (size_t s = 0; s < slice_timings_orig.size(); s++)
    {
        if (slice_timings_orig[s] == 0)
            MB++;
    }

    vnl_matrix<int> slspec(N_unique_times, MB);

    for (size_t sg = 0; sg < slice_timings.size(); sg++)
    {
        int col = 0;
        for (size_t s = 0; s < slice_timings_orig.size(); s++)
        {
            if (slice_timings_orig[s] == slice_timings[sg])
            {
                slspec(sg, col) = s;
                col++;
            }
        }
        if (col != MB)
        {
            float sg_time = slice_timings[sg];

            float mndst = 1E10;
            int min_ind = -1;
            for (size_t s = 0; s < slice_timings_orig.size(); s++)
            {
                float dist = fabs(sg_time - slice_timings_orig[s]);
                if (dist < mndst && dist != 0)
                {
                    mndst = dist;
                    min_ind = s;
                }
                if (dist == mndst)
                {
                    if ((int)s > slspec(sg, col - 1))
                    {
                        mndst = dist;
                        min_ind = s;
                    }
                }
            }
            slspec(sg, col) = min_ind;
            col++;
        }
    }
    return slspec;
}


// Compute slspec from JSON metadata and number of slices.
// Handles three cases:
//   1. SliceTiming present -> ParseJSONForSliceTiming
//   2. MultibandAccelerationFactor present -> regular interleaving
//   3. Neither -> single-band sequential identity
inline vnl_matrix<int> ComputeSlspec(json cjson, int Nslices)
{
    if (cjson.contains("SliceTiming") && !cjson["SliceTiming"].is_null())
    {
        return ParseJSONForSliceTiming(cjson);
    }
    else if (cjson.contains("MultibandAccelerationFactor") &&
             !cjson["MultibandAccelerationFactor"].is_null())
    {
        int MB = cjson["MultibandAccelerationFactor"];
        int Nexc = Nslices / MB;
        vnl_matrix<int> slspec(Nexc, MB);
        slspec.fill(0);
        for (int k = 0; k < Nslices; k++)
        {
            int r = k % Nexc;
            int c = k / Nexc;
            slspec(r, c) = k;
        }
        return slspec;
    }
    else
    {
        vnl_matrix<int> slspec(Nslices, 1);
        for (int k = 0; k < Nslices; k++)
            slspec(k, 0) = k;
        return slspec;
    }
}

#endif
