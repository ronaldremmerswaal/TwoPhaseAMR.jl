# Code check 02: the covered-neighbour rule on a grid-aligned solid boundary (the leak of the thesis' sum).
#
# Label:  eq:interface:phase-aperture / deviation "Open apertures" (07_cutcells.tex:117-125)
# Claim:  07_cutcells.tex:96-100: "the covered rule closes faces on a grid-aligned solid boundary, where both cells are
#         pure and the average would otherwise be 1/2".  With the thesis' definition a^o = a^liq + a^gas and (2.41) the
#         open aperture of such a face is 1/2 (covered cell contributes 0, open cell its trace 1, times 1/2).
# Code:   src/embedded/geometry.jl:140-156 (_interior_open_aperture), forces.jl:313 (_face_aperture), transport_geometry.jl.
# Method: 8x8 cells, solid {y <= 3} (grid line), liquid pool 3 < y <= 5 (grid line): all cells pure.  Faces between the
#         covered row j = 3 and the liquid row j = 4: code (a, liquid, gas) = (0, 0, 0) while (2.41) gives (1/2, 1/2, 0);
#         the liquid surface faces (rows 5/6) are pure and closed/open exactly; faces next to the covered row on the
#         gas side give (0,0,0) vs (1/2 trace, 0, 1/2 trace) when a gas cell is above a covered cell.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-phase-aperture/scripts/code_02_covered_neighbour.jl
#         (from the repository root)

import TwoPhaseAMR
using Test
const AMR = TwoPhaseAMR
isdefined(Main, :film_fixture) || include(joinpath(@__DIR__, "code_helpers.jl"))

@testset "grid-aligned solid boundary: covered rule closes the face, the thesis' sum leaves 1/2" begin
    mesh = AMR.BlockTreeMesh(; base_mesh = (8, 8), block_size = (8, 8), extent = (8.0, 8.0), periodic = (false, false))
    geometry = AMR.GeometryField(mesh, (x, y) -> y - 3.0)
    field = AMR.InterfaceField(mesh, (x, y) -> y - 5.0; geometry, contact_angle = nothing, embedded_contact_angle = nothing)
    found = Dict(:liquid_over_solid => 0, :gas_over_solid => 0)
    for (index, face) in pairs(mesh.faces)
        face.axis == 2 || continue
        covered_minus = AMR._get(geometry.open_fraction, face.minus) == 0
        covered_plus = AMR._get(geometry.open_fraction, face.plus) == 0
        (covered_minus != covered_plus) || continue
        liquid, gas = AMR._phase_face_apertures(field, index)
        a = geometry.aperture[index]
        th = thesis_apertures(field, geometry, face)
        @test a == 0 && liquid == 0 && gas == 0                              # the covered rule
        open_cell = covered_minus ? face.plus : face.minus
        pure_liquid = AMR._get(field.conditional_fraction, open_cell) == 1
        @test th.open == 0.5                                                 # the thesis' sum: (1/2)(0 + 1)
        @test th.liquid == (pure_liquid ? 0.5 : 0.0) && th.gas == (pure_liquid ? 0.0 : 0.5)
        found[pure_liquid ? :liquid_over_solid : :gas_over_solid] += 1
    end
    println("  faces closed by the covered rule: $(found[:liquid_over_solid]) under liquid, $(found[:gas_over_solid]) under gas; thesis aperture 1/2 on each")
    @test found[:liquid_over_solid] == 8 && found[:gas_over_solid] == 0
end

@testset "pool above the wall, gas above the pool, wall inside the gas: gas over solid" begin
    mesh = AMR.BlockTreeMesh(; base_mesh = (8, 8), block_size = (8, 8), extent = (8.0, 8.0), periodic = (false, false))
    geometry = AMR.GeometryField(mesh, (x, y) -> y - 3.0)
    field = AMR.InterfaceField(mesh, (x, y) -> y - 1.0e-3; geometry, contact_angle = nothing, embedded_contact_angle = nothing)   # no liquid above the wall
    n = 0
    for (index, face) in pairs(mesh.faces)
        face.axis == 2 || continue
        (AMR._get(geometry.open_fraction, face.minus) == 0) != (AMR._get(geometry.open_fraction, face.plus) == 0) || continue
        liquid, gas = AMR._phase_face_apertures(field, index)
        th = thesis_apertures(field, geometry, face)
        @test geometry.aperture[index] == 0 && liquid == 0 && gas == 0
        @test th.open == 0.5 && th.gas == 0.5 && th.liquid == 0.0
        n += 1
    end
    @test n == 8
end
