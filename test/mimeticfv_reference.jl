# Optional cross-package validation. This file is intentionally not included by
# `runtests.jl`, because MimeticFV is a development reference rather than a
# package dependency. Run with both project environments on JULIA_LOAD_PATH;
# see `test/reference-validation.md`.

using Test
import Meshes
import MimeticFV
import TwoPhaseAMR
using Unitful

const N = 8
const DT = 0.005

initial_value(x, y) = exp(sinpi(2x) + 0.5cospi(2y))

# Divergence-free polynomial velocity with zero normal velocity on every
# physical boundary. Its tangential dependence on each face is linear, so the
# MimeticFV face quadrature and TwoPhaseAMR midpoint sample agree to roundoff.
function reference_velocity(x, y)
    return (x * (1 - x) * (1 - 2y),
            -(1 - 2x) * y * (1 - y))
end

old = [initial_value((i - 0.5) / N, (j - 0.5) / N) for i in 1:N, j in 1:N]

mimetic_mesh = MimeticFV.PolygonMesh(
    Meshes.RectilinearGrid(range(0.0, 1.0; length=N + 1),
                           range(0.0, 1.0; length=N + 1)),
)
mimetic_scalar = MimeticFV.Form(vec(old), mimetic_mesh)
mimetic_velocity = MimeticFV.Form(mimetic_mesh) do x, y
    ux, uy = reference_velocity(ustrip(x / u"m"), ustrip(y / u"m"))
    [ux * u"m/s", uy * u"m/s"]
end
mimetic_result, = MimeticFV.advect(
    mimetic_velocity, DT * u"s", mimetic_scalar;
    flux_mode=:upwind, total_flux_mode=:raw,
)

tree_mesh = TwoPhaseAMR.BlockTreeMesh(
    block_size=(N, N), periodic=(false, false),
)
tree_scalar = TwoPhaseAMR.CellField(tree_mesh)
copyto!(TwoPhaseAMR.interior(tree_scalar, 1), old)
tree_velocity = TwoPhaseAMR.FaceVelocity(
    tree_mesh, (x, y, time, axis) -> reference_velocity(x, y)[axis],
)
workspace = TwoPhaseAMR.AdvectionWorkspace(tree_scalar)
TwoPhaseAMR.advect!(tree_scalar, tree_velocity, DT, workspace)

@testset "MimeticFV uniform :upwind reference" begin
    @test vec(collect(TwoPhaseAMR.interior(tree_scalar, 1))) ≈
          mimetic_result.data rtol=3e-14 atol=3e-14
end
