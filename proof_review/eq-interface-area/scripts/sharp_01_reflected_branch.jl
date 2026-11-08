# Sharpness 01: what does "the reflected branch avoids cancellation near full cells" mean?
#
# Label:  eq:interface:area
# Claim:  04_interface.tex:155-157: "the reflected branch avoids cancellation near full cells";
#         code comment interface_field.jl:239 "Symmetry avoids cancellation near full cells".
# Method: area fraction of the reference cell near full cells (distance zp = w1+w2-z to the far
#         corner from 0.5 w_< down to 1e-14 w_<, which keeps z in the third regime z >= w_>) for
#         strongly anisotropic widths w_</w_> = 1 ... 1e-8, against exact polygon clipping in
#         256-bit BigFloat, for
#           (a) the package (reflected branch),
#           (b) the explicit third regime 1 - (w1+w2-z)^2/(2 w1 w2) (what the reflection reproduces),
#           (c) the textbook inclusion--exclusion CDF
#               (z^2 - (z-w1)_+^2 - (z-w2)_+^2 + (z-w1-w2)_+^2)/(2 w1 w2).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-area/scripts/sharp_01_reflected_branch.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo
setprecision(BigFloat, 256)

third_regime(w1, w2, z) = 1 - (w1 + w2 - z)^2 / (2w1 * w2)
function inclusion_exclusion(w1, w2, z)
    pos(x) = max(x, zero(x))
    return (z^2 - pos(z - w1)^2 - pos(z - w2)^2 + pos(z - w1 - w2)^2) / (2w1 * w2)
end

@testset "reflected branch versus the explicit third regime and inclusion-exclusion" begin
    println("  w_</w_>   zp/w_<    |code-ref|    |third regime-ref|   |incl.-excl.-ref|")
    err_code = err_third = err_ie_aniso = 0.0
    for ratio in (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8), zp_rel in (0.5, 1.0e-2, 1.0e-6, 1.0e-10, 1.0e-14)
        e_code = e_third = e_ie = 0.0
        w2 = 1 / (1 + ratio); w1 = ratio / (1 + ratio)             # w1 + w2 = 1 on the reference cell
        for (n1, n2) in ((w2, w1), (w1, w2), (-w2, w1), (w1, -w2))
            a = 0.5 - zp_rel * w1                                  # plane value z = w1 + w2 - zp_rel w_< (up to rounding)
            ref = Float64(
                plane_area_reference(BigFloat(n1), BigFloat(n2), BigFloat(a), BigFloat(-0.5), BigFloat(0.5), BigFloat(-0.5), BigFloat(0.5))
            )
            shift = min(n1 * -0.5, n1 * 0.5) + min(n2 * -0.5, n2 * 0.5)
            z = a - shift                                          # the plane value the closed form uses
            e_code = max(e_code, abs(AMR._plane_fraction(n1, n2, a) - ref))
            e_third = max(e_third, abs(third_regime(abs(n1), abs(n2), z) - ref))
            e_ie = max(e_ie, abs(inclusion_exclusion(abs(n1), abs(n2), z) - ref))
        end
        @printf("  %-8.0e  %-8.0e  %-12.3e  %-18.3e  %.3e\n", ratio, zp_rel, e_code, e_third, e_ie)
        err_code = max(err_code, e_code); err_third = max(err_third, e_third)
        ratio <= 1.0e-4 && (err_ie_aniso = max(err_ie_aniso, e_ie))
    end
    println("  worst: code $(err_code), explicit third regime $(err_third); inclusion-exclusion for w_</w_> <= 1e-4: $(err_ie_aniso)")
    # the package's evaluation is as accurate as the explicit third regime ...
    @test err_code <= 8eps()
    @test err_third <= 8eps()
    # ... while the inclusion-exclusion form loses ~ eps w_>/w_< (cancellation of terms of size w_>^2)
    @test err_ie_aniso > 1.0e3 * err_code
end
