# Code check 05: the two force maps of the two-velocity step and the balance of
# the hydrostatic rest state and of a constant-curvature drop through CMDGFM
# (issue #76, re-verified on current main after the removal of the mimetic
# projection in #137).
#
# Label:  eq:forces:two-velocity, eq:forces:map  (issue #76)
# Source: docs/techref/chapters/08_forces.tex (Sec. "two velocities"),
#         src/two_phase/forces.jl (_weighted_jump_force!, PoissonOperator),
#         src/two_phase/projection/common.jl (_projection_pressure_operator,
#         _cut_cell_weighted_gradient!).
#
# Claims checked, on the fixtures of docs/techref/figures/forces.jl:
#   M1  the one-velocity cut-cell map and the projection's map are congruent,
#       R^1 = A^{1/2} R^tv A^{1/2} with A = diag(aperture), entrywise
#       R^1_fg = sqrt(a_f a_g) R^tv_fg, and R^1 = A R^tv exactly at a face whose
#       row of the inverse metric K is diagonal;
#   M2  E = R^1 J - R^tv (D p*) with p* the least-squares cell function, i.e.
#       the part of the one-velocity increment that no gradient of the
#       projection's map can remove: ~1e-11 on every conforming fixture and
#       with no solid, 1.1e-2 / 1.2e-2 on the composite mesh with a solid and
#       3.1e-3 for the drop on the cut mesh;
#   P1  the code's increment (the projection's cached map, as _phase_force_substep!
#       evaluates it) is removed by the projection to <= 1e-9 of the increment;
#   P2  the former one-velocity increment of the same jumps leaves 1.5e-2 of
#       max|R^1 J| (2.9e-2 of dt|g|) on the composite mesh with a solid and
#       1.3e-2 for the drop on the cut mesh.
#
# Run: julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-two-velocity/scripts/code_05_current_map.jl
#      (from the repository root; needs the root Manifest.toml)

using Test, LinearAlgebra, SparseArrays, Random
using TwoPhaseAMR
const AMR = TwoPhaseAMR

const BOX = 8.0
const DROP = (; centre = (4.3, 4.1), radius = 2.23)
const DENSITY = (1000.0, 1.0)
const SIGMA = 0.7
const DT = 0.01
const GRAVITY = (0.0, -1.0)

solid_low(x, y) = y - 0.9 - 0.1x
solid_high(x, y) = 7.1 - 0.1x - y
drop_levelset(x, y) = hypot(x - DROP.centre[1], y - DROP.centre[2]) - DROP.radius

function mesh_of(kind::Symbol)
    return BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), extent = (BOX, BOX),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) ->
        kind != :uniform && key.level == 0 && key.i >= 2,
    )
end

function plane_field(kind::Symbol; solid = :none)
    mesh = mesh_of(kind)
    levelset = (x, y) -> y - 4.6
    solid == :none && return InterfaceField(mesh, levelset; lipschitz = 1)
    shape = solid == :liquid ? solid_low : solid_high
    geometry = GeometryField(mesh, shape; lipschitz = hypot(1, 0.1))
    return InterfaceField(
        mesh, levelset; geometry, contact_angle = pi / 2, lipschitz = 1,
    )
end

function drop_field(kind::Symbol)
    mesh = mesh_of(kind == :uniform ? :uniform : :amr)
    kind == :cut || return InterfaceField(mesh, drop_levelset; lipschitz = 1)
    geometry = GeometryField(mesh, solid_low; lipschitz = hypot(1, 0.1))
    return InterfaceField(
        mesh, drop_levelset; geometry, contact_angle = pi / 2, lipschitz = 1,
    )
end

one_velocity_operator(field) = isnothing(field.geometry) ?
    PoissonOperator(field; density = DENSITY) :
    PoissonOperator(field, field.geometry; density = DENSITY)

"""The operator of the force stage of the two-velocity step (PR #81): the
projection's cached map, A/rho with a solid, the one-velocity operator
without one."""
function code_force_operator(field)
    solver = CMDGFMProjectionSolver(field; density = DENSITY)
    return AMR._phase_force_pressure_operator!(
        solver, CMDGFMProjectionWorkspace(solver),
    )
end

function mapped(operator, field, jumps)
    destination = FaceVelocity(field.mesh)
    AMR._weighted_jump_force!(
        destination, operator, EllipticWorkspace(operator), jumps, field.geometry,
    )
    return copy(destination.interior)
end

two_maxabs(phases) = maximum(
    maximum(value -> isnan(value) ? 0.0 : abs(value), phase.interior)
        for phase in phases
)

"""Project the phase increments and return the leftover over the increment."""
function projected_leftover(field, force!, operator)
    acceleration = FaceVelocity(field.mesh)
    force!(acceleration, field, operator)
    scale = DT * maximum(abs, acceleration.interior)
    phases = (FaceVelocity(field.mesh), FaceVelocity(field.mesh))
    for phase in phases
        phase.interior .= DT .* acceleration.interior
    end
    AMR._mark_inactive_phase_velocities!(phases, field)
    solver = CMDGFMProjectionSolver(field; density = DENSITY)
    stats = project!(
        phases, CellField(field.mesh), solver, DT,
        CMDGFMProjectionWorkspace(solver);
        tolerance = 1.0e-12, max_iterations = 4000,
    )
    @assert stats.converged
    return two_maxabs(phases) / scale
end

gravity_force!(acceleration, field, operator) = gravity!(
    acceleration, field, operator, EllipticWorkspace(operator);
    density = DENSITY, gravity = GRAVITY,
)

function tension_force!(acceleration, field, operator)
    return surface_tension!(
        acceleration, field, operator, EllipticWorkspace(operator);
        sigma = SIGMA, formulation = CSF(), curvature = 1 / DROP.radius,
    )
end

"""E = R^1 J - R^tv (D p*), the residual of the one-velocity increment outside
the range of the projection's pressure gradient, relative to max|R^1 J|."""
function balance_residual(field, force!)
    mesh = field.mesh
    one = one_velocity_operator(field)
    tv = code_force_operator(field)
    jumps = FaceVelocity(mesh)
    force!(jumps, field, one)
    r1 = copy(jumps.interior)
    # Columns of R^tv (D .): the common cut-cell weighted gradient.
    columns = Vector{Vector{Float64}}()
    potential = CellField(mesh)
    for ref in AMR._cell_references(mesh)
        for block in eachindex(potential.blocks)
            fill!(interior(potential, block), 0.0)
        end
        fill_halo!(potential)
        AMR._set!(potential, 1.0, ref)
        gradient = FaceVelocity(mesh)
        if isnothing(field.geometry)
            weighted_gradient!(gradient, potential, tv, EllipticWorkspace(tv))
        else
            AMR._cut_cell_weighted_gradient!(
                gradient, potential, tv, EllipticWorkspace(tv), field.geometry,
            )
        end
        push!(columns, copy(gradient.interior))
    end
    projector = hcat(columns...)
    pressure = projector \ r1
    residual = r1 .- projector * pressure
    return maximum(abs, residual) / maximum(abs, r1)
end

@testset "eq-forces-two-velocity: current maps and balances (#76)" begin
    Random.seed!(20261112)
    @testset "M1 map congruence" begin
        for (kind, solid) in ((:uniform, :gas), (:amr, :gas), (:amr, :liquid))
            field = plane_field(kind; solid)
            one, tv = one_velocity_operator(field), code_force_operator(field)
            aperture = field.geometry.aperture
            jumps = randn(length(field.mesh.faces))
            r1 = mapped(one, field, jumps)
            rtv = mapped(tv, field, jumps)
            congruent = mapped(tv, field, sqrt.(aperture) .* jumps) .* sqrt.(aperture)
            @test maximum(abs, r1 .- congruent) <= 1.0e-14 * maximum(abs, r1)
            metric = tv.gradient_operator.inverse_metric
            diagonal = metric isa AMR.DiagonalInverseFaceMetric ?
                trues(length(aperture)) : begin
                    matrix = metric.matrix
                    [
                        sum(!iszero, nonzeros(matrix)[nzrange(matrix, i)]) == 1
                        for i in eachindex(aperture)
                    ]
                end
            rows = findall(diagonal)
            @test !isempty(rows)
            @test maximum(abs, (r1 .- aperture .* rtv)[rows]) <=
                1.0e-14 * maximum(abs, r1)
            if kind == :uniform
                @test all(diagonal)
                @test maximum(abs, r1 .- aperture .* rtv) <= 1.0e-14 * maximum(abs, r1)
            end
        end
    end

    @testset "M2 residual outside the projection's range" begin
        for solid in (:none, :gas, :liquid), kind in (:uniform, :amr)
            residual = balance_residual(plane_field(kind; solid), gravity_force!)
            if kind == :amr && solid != :none
                @test residual > 1.0e-3
            else
                @test residual <= 1.0e-8
            end
        end
        for kind in (:uniform, :amr)
            @test balance_residual(drop_field(kind), tension_force!) <= 1.0e-8
        end
        @test balance_residual(drop_field(:cut), tension_force!) > 1.0e-3
    end

    @testset "P1/P2 increments through both maps" begin
        for solid in (:none, :gas, :liquid), kind in (:uniform, :amr)
            field = plane_field(kind; solid)
            current = projected_leftover(field, gravity_force!, code_force_operator(field))
            former = projected_leftover(field, gravity_force!, one_velocity_operator(field))
            @test current <= 1.0e-9
            if kind == :amr && solid != :none
                @test former > 1.0e-3
            else
                @test former <= 1.0e-8
            end
        end
        for kind in (:uniform, :amr)
            field = drop_field(kind)
            # Without a solid the two maps are bitwise equal, so this is the
            # code's increment too.
            @test projected_leftover(
                field, tension_force!, one_velocity_operator(field),
            ) <= 1.0e-9
        end
        cut = drop_field(:cut)
        @test projected_leftover(cut, tension_force!, code_force_operator(cut)) <= 1.0e-9
        @test projected_leftover(cut, tension_force!, one_velocity_operator(cut)) > 1.0e-3
    end
end
