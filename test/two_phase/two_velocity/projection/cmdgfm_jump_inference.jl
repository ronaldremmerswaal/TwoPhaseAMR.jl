using Test, TwoPhaseAMR, LinearAlgebra
include("../helpers.jl")

const CJI = TwoPhaseAMR

# Single projections on the unit square, periodic in x with walls at y = 0, 1,
# n × n cells, the liquid below a graph (the laboratory of issue #171). The
# predictor is `phase_velocity(x, y, axis, phase)`; `wall_flux = false` sets
# its wall values to zero.
function _jump_inference_projection(
        n, levelset, phase_velocity; inference, density = (1.0, 1.0),
        dt = 1.0, wall_flux = true, tolerance = 1.0e-9,
    )
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (8, 8), periodic = (true, false),
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    solver = CMDGFMProjectionSolver(interface; density, jump_inference = inference)
    work = CMDGFMProjectionWorkspace(solver)
    velocity = ntuple(
        phase -> FaceVelocity(mesh, (x, y, time, axis) -> phase_velocity(x, y, axis, phase)),
        2,
    )
    wall_flux || foreach(phase -> fill!(velocity[phase].boundary, 0.0), 1:2)
    stats = project!(
        velocity, CellField(mesh), solver, dt, work; tolerance, max_iterations = 4000,
    )
    return (; mesh, interface, velocity, stats, data = work.pressure_cache.face_data)
end

_phase_jump(velocity) = velocity[2].interior .- velocity[1].interior

function _largest_jump(result; window = x -> true)
    jump = _phase_jump(result.velocity)
    return maximum(
        index -> isfinite(jump[index]) && window(result.mesh.faces[index].center[1]) ?
            abs(jump[index]) : 0.0,
        eachindex(jump),
    )
end

_wavy(y0, amplitude) = (x -> y0 + amplitude * cospi(2x), x -> -2pi * amplitude * sinpi(2x))
_triangle(y0, slope) = (
    x -> y0 + slope * (0.25 - abs(mod(x, 1.0) - 0.5)),
    x -> -slope * sign(mod(x, 1.0) - 0.5),
)
# Faces within 0.1 of the kinks of the triangle are not measured.
_away_from_kinks(x) = min(abs(x - 0.5), x, 1 - x) > 0.1

@testset "CMDGFM jump inference option" begin
    @test CMDGFMProjection().jump_inference isa LinearJumpInference
    thesis = CMDGFMProjection(jump_inference = ConstantJumpInference())
    @test thesis.jump_inference isa ConstantJumpInference
    @test thesis.preconditioner === :block
    @test CMDGFMProjection(:ilu, (;), ConstantJumpInference()).jump_inference isa
        ConstantJumpInference
    @test CMDGFMProjection(:ilu, (;)).jump_inference isa LinearJumpInference
    @test_throws ArgumentError CMDGFMProjection(jump_inference = :linear)
    @test_throws ArgumentError CMDGFMProjection(jump_inference = nothing)
    model = TwoVelocityModel(pressure_projection = thesis)
    @test model.pressure_projection.jump_inference isa ConstantJumpInference
    @test TwoVelocityModel().pressure_projection.jump_inference isa LinearJumpInference

    mesh = BlockTreeMesh(block_size = (16, 16), periodic = (true, false))
    interface = InterfaceField(mesh, (x, y) -> y - 0.5 - 0.02cospi(2x); lipschitz = 1)
    @test CMDGFMProjectionSolver(interface).jump_inference isa LinearJumpInference
    @test_throws ArgumentError CMDGFMProjectionSolver(interface; jump_inference = :linear)

    # The model's option reaches the projection solver of a flow.
    flow = FlowSimulation(
        mesh, TwoPhase(interface; density = (10.0, 1.0), dynamic_viscosity = (0.0, 0.0));
        velocity_model = model, boundary = FreeSlip(),
        velocity = ((x, y, time, axis) -> 0.0, (x, y, time, axis) -> 0.0), dt = 1.0e-3,
    )
    state = CJI._two_velocity_flow_workspace!(flow.workspace.two_phase, flow.system.two_phase)
    @test state.projection_solver.jump_inference isa ConstantJumpInference

    # The thesis rows report no extension rows and no fallbacks.
    shear = (x, y, axis, phase) -> axis == 1 ? tanh((y - 0.5) / 0.1) : 0.0
    levelset = (x, y) -> y - _wavy(0.5, 0.01)[1](x)
    constant = _jump_inference_projection(16, levelset, shear; inference = ConstantJumpInference())
    @test constant.stats.converged
    @test all(iszero, values(constant.stats.jump_inference))
    extended = _jump_inference_projection(16, levelset, shear; inference = LinearJumpInference())
    @test extended.stats.converged
    @test extended.stats.jump_inference.extrapolated_rows > 0
    @test contains(sprint(show, extended.stats), "jump_inference=")
end

@testset "Linear jump inference: smooth shear across tilted and curved interfaces" begin
    # A single-valued, solenoidal shear u = U(y) with a resolved layer is a
    # fixed point of the exact projection. The constant inference creates a
    # jump of order 2U'h; the linear one of order 2h³U'''. On the wavy and the
    # slope-0.06 interfaces the rows see crossing ratios w/U ≤ 0.028 (largest
    # at n = 128), below the lower end 1/32 of the crossing-speed ramp, so they
    # keep θ = 1; with the lower end 1/48 the slope-0.06 jump at n = 128 rises
    # from 0.0076 to 0.017 and the comparison with the constant rows fails.
    delta = 0.05
    shear = (x, y, axis, phase) -> axis == 1 ? 0.5tanh((y - 0.5) / delta) : 0.0
    for (graph, window, crossing) in (
            (_wavy(0.5, 0.01), x -> true, false),
            (_triangle(0.5, 0.06), _away_from_kinks, false),
            (_triangle(0.5, 0.3), _away_from_kinks, true),
        )
        levelset = (x, y) -> y - graph[1](x)
        created = Dict{Symbol, Vector{Float64}}()
        for (name, inference) in (
                (:constant, ConstantJumpInference()), (:linear, LinearJumpInference()),
            )
            created[name] = map([32, 64, 128]) do n
                result = _jump_inference_projection(n, levelset, shear; inference)
                @test result.stats.converged
                if inference isa LinearJumpInference
                    counts = result.stats.jump_inference
                    @test counts.extrapolated_rows > 0
                    @test counts.level_fallbacks == counts.solid_fallbacks == 0
                end
                _largest_jump(result; window)
            end
        end
        linear = created[:linear]
        @test all(<(0), diff(linear))
        if crossing
            # NUMERICS CONCERN (#171): the shear u = U(y) e_x crosses the
            # interface of slope 0.3 at the normal speed |U n_x|, i.e. at
            # w/U ≈ |n_x| = 0.29 > 3/32 relative to the local mixture speed:
            # kinematically this flow does move the interface through the
            # faces, so the crossing-speed limiter correctly sees a crossing
            # and falls back to the one-donor rule. The created jump is 0.60,
            # 0.29, 0.171 at n = 32, 64, 128 (published rows 0.50, 0.29, 0.152),
            # first order, where the unlimited extrapolation gives 0.26,
            # 0.059, 0.010. The limiter depends on the crossing speed only and
            # cannot tell a smooth field (where the extrapolation is safe in a
            # single projection) from the noisy wake of a sweeping interface;
            # unchanged by freezing θ and by the dominance cap (same values).
            @test_broken log2(linear[2] / linear[3]) >= 2
            @test_broken log2(linear[1] / linear[3]) / 2 >= 2
            @test_broken linear[end] <= 0.1 * created[:constant][end]
        else
            @test log2(linear[2] / linear[3]) >= 2
            @test log2(linear[1] / linear[3]) / 2 >= 2
            @test linear[end] <= 0.1 * created[:constant][end]
        end
    end
end

@testset "Linear jump inference: hydrostatic rest on a tilted interface" begin
    # Gravity acting on fluids at rest, u* = -g dt e_y without wall flux: the
    # projection returns rest. Equal densities on the tilted triangle (the jump
    # rows must not create a jump from a single-valued predictor), and a
    # density contrast on a flat interface between grid lines.
    dt = 0.01
    gravity = (x, y, axis, phase) -> axis == 2 ? -dt : 0.0
    for (graph, density) in (
            (_triangle(0.5 + 0.37 / 32, 0.3), (1.0, 1.0)),
            (_wavy(0.5 + 0.37 / 32, 0.0), (1000.0, 1.0)),
        )
        result = _jump_inference_projection(
            32, (x, y) -> y - graph[1](x), gravity; density, dt,
            inference = LinearJumpInference(), wall_flux = false, tolerance = 1.0e-11,
        )
        @test result.stats.converged
        largest = maximum(
            value -> isfinite(value) ? abs(value) : 0.0,
            vcat(result.velocity[1].interior, result.velocity[2].interior),
        )
        @test largest <= 1.0e-8 * dt
        @test _phase_storage_contract(result.velocity, result.interface)
    end
end

@testset "Linear jump inference: flat interface and single-valued gradients are exact" begin
    # A continuous shear across a flat interface between grid lines: every
    # x-face of the band is exactly tangential and its row is imposed as an
    # increment of the face's own jump, so no jump is created.
    delta = 0.05
    shear = (x, y, axis, phase) -> axis == 1 ? 0.5tanh((y - 0.5) / delta) : 0.0
    for n in (32, 64)
        result = _jump_inference_projection(
            n, (x, y) -> y - 0.5 - 0.37 / n, shear; inference = LinearJumpInference(),
        )
        @test result.stats.converged
        @test result.stats.jump_inference.tangential_rows == n
        @test _largest_jump(result) <= 1.0e-12
    end
    # A single-valued gradient u* = -G φ is removed exactly: every row, with
    # its frozen slope weight, is homogeneous in the projected state.
    for levelset in (
                n -> (x, y) -> y - _wavy(0.5 + 0.37 / n, 0.02)[1](x),
                n -> (x, y) -> y - _triangle(0.5 + 0.37 / n, 0.3)[1](x),
                n -> (x, y) -> hypot(x - 0.5 - 0.21 / n, y - 0.5 - 0.37 / n) - 0.25,
            ), n in (32, 64)
        mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic = (true, false))
        interface = InterfaceField(mesh, levelset(n); lipschitz = 1)
        solver = CMDGFMProjectionSolver(interface)
        work = CMDGFMProjectionWorkspace(solver)
        phi = CellField(mesh, (x, y) -> cospi(2x) * cospi(y))
        velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
        for (index, face) in pairs(mesh.faces)
            gradient = (CJI._get(phi, face.plus) - CJI._get(phi, face.minus)) * n
            velocity[1].interior[index] = velocity[2].interior[index] = -gradient
        end
        foreach(phase -> fill!(velocity[phase].boundary, 0.0), 1:2)
        scale = maximum(abs, velocity[1].interior)
        stats = project!(
            velocity, CellField(mesh), solver, 1.0, work;
            tolerance = 1.0e-9, max_iterations = 4000,
        )
        @test stats.converged
        @test stats.jump_inference.extrapolated_rows > 0
        largest = maximum(
            value -> isfinite(value) ? abs(value) : 0.0,
            vcat(velocity[1].interior, velocity[2].interior),
        )
        @test largest <= 1.0e-10 * scale
    end
end

@testset "Linear jump inference: crossing-speed limiter" begin
    # A uniform translation u = (1, 0) of both phases through a circle: the
    # interface crosses each face at w/U = |n_x|, so the slope weight is one
    # minus the smoothstep of |n_x| over the ramp [1/32, 3/32]; at rest every
    # row keeps the linear extrapolation. Exchanging the phase labels (the
    # complement of the circle, densities swapped) gives the same weights.
    n = 32
    mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic = (true, true))
    lower, upper = CJI._CMDGFM_CROSSING_RAMP
    normal_inference = LinearJumpInference()
    sliding_inference = LinearJumpInference(; limiter = :sliding)
    @test_throws ArgumentError LinearJumpInference(; limiter = :unknown)
    @test (lower, upper) == (1 / 32, 3 / 32)
    ramp(r) = (s = clamp((r - lower) / (upper - lower), 0.0, 1.0); 1 - s^2 * (3 - 2s))
    weights = map((1, -1)) do label
        interface = InterfaceField(
            mesh, (x, y) -> label * (hypot(x - 0.47, y - 0.52) - 0.27); lipschitz = 1,
        )
        density = label == 1 ? (1000.0, 1.0) : (1.0, 1000.0)
        solver = CMDGFMProjectionSolver(interface; density)
        work = CMDGFMProjectionWorkspace(solver)
        data = CJI._projection_face_data(solver, work)
        stencils = filter(stencil -> stencil.resolved, CJI._cmdgfm_jump_stencils(solver, data))
        @test count(stencil -> stencil.gas_extrapolation_face > 0, stencils) > 8
        # The cap (1 - s)/s of the slope weight that keeps the row weakly
        # diagonally dominant (s: cross mass fractions of the first donors).
        function cap(stencil)
            gas, liquid = stencil.gas_inference_face, stencil.liquid_inference_face
            s = data.liquid_mass[gas] / data.total_mass[gas] +
                data.gas_mass[liquid] / data.total_mass[liquid]
            return s > 0 ? (1 - s) / s : Inf
        end
        rest = (FaceVelocity(mesh), FaceVelocity(mesh))
        @test all(
            stencil -> stencil.gas_extrapolation_face == 0 ? stencil.slope == 1 :
                stencil.slope == min(1.0, cap(stencil)),
            CJI._cmdgfm_limit_extrapolation!(copy(stencils), data, rest, interface, normal_inference),
        )
        translation = ntuple(_ -> FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0), 2)
        limited = CJI._cmdgfm_limit_extrapolation!(copy(stencils), data, translation, interface, normal_inference)
        slopes = Dict{Int, Float64}()
        for stencil in limited
            stencil.gas_extrapolation_face > 0 || continue
            normal = CJI._cmdgfm_face_normal(interface, mesh.faces[stencil.face])
            @test stencil.slope ≈ min(ramp(abs(normal[1])), cap(stencil)) atol = 1.0e-12
            slopes[stencil.face] = stencil.slope
        end
        @test any(<(1), values(slopes)) && any(==(1), values(slopes))
        # The phases sliding past each other: the heavy phase at rest, the
        # light one moving with u = (1, 0). The mixture of a face with heavy
        # mass hardly moves, so the interface normal speed |ū_x n_x| of an
        # x-face row stays below |n_x|, but the sliding speed
        # |u_g,x - u_l,x| |n_x| = |n_x| and U = 1 (the pure light second donor)
        # give the x-face rows the weights of the translation. On some of them
        # the normal speed alone would leave a larger weight.
        light = label == 1 ? 2 : 1
        sliding = ntuple(
            phase -> FaceVelocity(
                mesh, (x, y, time, axis) -> phase == light && axis == 1 ? 1.0 : 0.0,
            ), 2,
        )
        incidence = CJI._cmdgfm_incident_faces(mesh)
        normal_rows = CJI._cmdgfm_limit_extrapolation!(copy(stencils), data, sliding, interface, normal_inference)
        normal_slopes = Dict(stencil.face => stencil.slope for stencil in normal_rows)
        x_rows = larger_alone = 0
        for stencil in CJI._cmdgfm_limit_extrapolation!(copy(stencils), data, sliding, interface, sliding_inference)
            stencil.gas_extrapolation_face > 0 && mesh.faces[stencil.face].axis == 1 || continue
            @test stencil.slope ≈ slopes[stencil.face] atol = 1.0e-12
            x_rows += 1
            normal = CJI._cmdgfm_face_normal(interface, mesh.faces[stencil.face])
            mixture = CJI._cmdgfm_mixture_vector(
                stencil.face, data, sliding[1].interior, sliding[2].interior, mesh, incidence,
            )
            alone = min(ramp(abs(mixture[1] * normal[1] + mixture[2] * normal[2])), cap(stencil))
            @test normal_slopes[stencil.face] ≈ alone atol = 1.0e-12
            alone > stencil.slope + 0.1 && (larger_alone += 1)
        end
        @test x_rows > 4
        @test larger_alone > 0
        slopes
    end
    @test keys(weights[1]) == keys(weights[2])
    @test all(face -> isapprox(weights[1][face], weights[2][face]; atol = 1.0e-12), keys(weights[1]))
end

@testset "Linear jump inference: a translating drop stays at the one-donor level" begin
    # Drop R = 0.2 carried at U = 1 through a periodic channel, density ratio
    # 1000, σ = 0.2, inviscid, n = 32, t = 1. Without the crossing-speed limiter
    # the second donors left behind by the sweeping interface form a double
    # integrator and max|u - U| grows to 1.09 (3.85 at n = 64); the published
    # rows give 0.176, the limited linear inference 0.133.
    function drift(inference)
        n = 32
        mesh = BlockTreeMesh(; block_size = (8, 8), base_mesh = (n, n), periodic = (true, false))
        centre = (0.5 + 0.13 / n, 0.5 + 0.29 / n)
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - centre[1], y - centre[2]) - 0.2; lipschitz = 1,
        )
        flow = FlowSimulation(
            mesh, TwoPhase(
                interface; dynamic_viscosity = (0.0, 0.0), density = (1.0, 1.0e-3),
                surface_tension = SurfaceTension(0.2),
            );
            velocity = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0,
            velocity_model = TwoVelocityModel(
                pressure_projection = CMDGFMProjection(; jump_inference = inference),
            ),
            boundary = FreeSlip(),
        )
        largest = 0.0
        while flow.time < 1 - 1.0e-12
            # step! throws if the projection does not converge.
            step!(flow, min(stable_timestep(flow), 1 - flow.time); tolerance = 1.0e-10, max_cycles = 4000)
            for (index, face) in pairs(mesh.faces)
                c = (CJI._conditional_fraction(interface, face.minus) + CJI._conditional_fraction(interface, face.plus)) / 2
                for (phase, fraction) in ((1, c), (2, 1 - c))
                    value = flow.velocity[phase].interior[index]
                    fraction > 0 && isfinite(value) || continue
                    largest = max(largest, abs(value - (face.axis == 1 ? 1.0 : 0.0)))
                end
            end
        end
        return largest
    end
    thesis, linear = drift(ConstantJumpInference()), drift(LinearJumpInference())
    @test thesis <= 0.25
    @test linear <= 1.1 * thesis
end

# u_l = (-1/2, 0), u_g = (1/2, 0) over y = y0 + eps cos(kx): the projection
# removes the normal jump U n_x and, with a density contrast, changes the
# sheet strength by U eps k [1/rho] cos(kx) / (T_g/rho_g + T_l/rho_l) to first
# order in eps (walls at y = 0, 1). Returns the mass-weighted rms deviation of
# the projected face jump from that sheet, relative to the linear scale U eps k.
function _sheet_error(n, density, inference; epsilon = 0.005, k = 2pi)
    sheet = (x, y, axis, phase) -> axis == 1 ? (phase == 1 ? -0.5 : 0.5) : 0.0
    y0 = 0.5 + 0.37 / n
    graph = _wavy(y0, epsilon)
    liquid_depth, gas_depth = tanh(k * y0), tanh(k * (1 - y0))
    rho_l, rho_g = density
    response = (1 / rho_g - 1 / rho_l) / (gas_depth / rho_g + liquid_depth / rho_l)
    result = _jump_inference_projection(
        n, (x, y) -> y - graph[1](x), sheet; density, inference,
    )
    jump = _phase_jump(result.velocity)
    numerator = denominator = 0.0
    for index in result.data.mixed
        face = result.mesh.faces[index]
        x = face.center[1]
        slope = graph[2](x)
        tangent = (1.0, slope) ./ hypot(1.0, slope)
        strength = tangent[1] + epsilon * k * response * cospi(2x)
        weight = result.data.relative_mass[index]
        numerator += weight * (jump[index] - strength * tangent[face.axis])^2
        denominator += weight
    end
    return (; error = sqrt(numerator / denominator) / (epsilon * k), result.stats)
end

@testset "Linear jump inference: a vortex sheet keeps its tangential jump" begin
    # The deviation stays below 5 % of the linear scale U eps k, i.e. below
    # 0.2 % of the sheet strength.
    for density in ((1.0, 1.0), (10.0, 1.0)), n in (32, 64)
        errors = map((ConstantJumpInference(), LinearJumpInference())) do inference
            sheet = _sheet_error(n, density, inference)
            @test sheet.stats.converged
            sheet.error
        end
        @test errors[2] <= 0.05
        density[1] == density[2] || @test errors[2] <= 0.5 * errors[1]
    end
end

@testset "Linear jump inference: near-tangential faces without a regular row" begin
    # At n = 128 the interface (amplitude 0.64 h) leaves x-faces with
    # |n_t| > |n_a| next to the crests without the two transverse mixed faces
    # of a regular row. Keeping their jump (common increment) misses the
    # density-driven change of the sheet strength there: 0.142 (thesis rows)
    # and 0.085 (normal rows restricted to |n_t| <= |n_a| only) at the density
    # ratio 1000; their beta = 0 inference row gives 0.010 (#171).
    for (density, bound) in (((10.0, 1.0), 0.01), ((1000.0, 1.0), 0.02))
        thesis = _sheet_error(128, density, ConstantJumpInference())
        linear = _sheet_error(128, density, LinearJumpInference())
        @test thesis.stats.converged && linear.stats.converged
        @test linear.error <= bound
        @test linear.error <= 0.2 * thesis.error
        @test linear.stats.jump_inference.tangential_rows > 2
    end
end

@testset "Linear jump inference: momentum, defects and fallbacks" begin
    # Momentum is conserved face by face for any gradient jump, also on the
    # normal rows; the composite mesh with a solid exercises the fallbacks.
    meshes = (
        (BlockTreeMesh(block_size = (16, 16), periodic = (true, true)), nothing),
        let mesh = BlockTreeMesh(
                block_size = (8, 8), max_level = 2,
                refine = (key, origin, extent) -> key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 1),
            )
            geometry = GeometryField(
                mesh, (x, y) -> y - 0.1 - 0.05x; lipschitz = hypot(1, 0.05),
            )
            (mesh, geometry)
        end,
    )
    for (mesh, geometry) in meshes, density in ((1.0, 1.0), (1000.0, 1.0))
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.45, y - 0.43) - 0.25; geometry, lipschitz = 1,
        )
        velocity = (
            FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? sinpi(2y) + 0.1cospi(2x) : -0.2sinpi(2x)),
            FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 0.3cospi(2y) : sinpi(2x)),
        )
        isnothing(geometry) || enforce_no_permeation!(velocity)
        solver = CMDGFMProjectionSolver(
            interface; density, jump_inference = LinearJumpInference(),
        )
        workspace = CMDGFMProjectionWorkspace(solver)
        tolerance = 1.0e-9
        stats = project!(
            velocity, CellField(mesh), solver, 0.02, workspace;
            tolerance, max_iterations = 4000,
        )
        @test stats.converged
        @test stats.mixture_volume_defect <= tolerance
        @test stats.normal_velocity_defect <= tolerance
        @test stats.momentum_identity_error <= 1024eps() * maximum(density)
        counts = stats.jump_inference
        @test counts.extrapolated_rows > 0
        @test counts.normal_rows > 0
        if isnothing(geometry)
            @test maximum(abs, stats.momentum_error) <= 1024eps() * maximum(density)
            @test counts.level_fallbacks == counts.solid_fallbacks == 0
        else
            @test counts.level_fallbacks > 0
            @test counts.solid_fallbacks > 0
        end
        @test _phase_storage_contract(velocity, interface)
    end
end

# Face-interpolated normal jump n_a [u]_f + n_t sum_h w_h [u]_h with the analytic
# normal of the graph and the weights c(1 - c) of the mixed transverse faces.
function _interpolated_normal_jump(result, slope_of; window = x -> true)
    mesh, data = result.mesh, result.data
    jump = _phase_jump(result.velocity)
    n = round(Int, sqrt(CJI.ncells(mesh)))
    key(center) = (mod(round(Int, 2n * center[1]), 2n), round(Int, 2n * center[2]))
    lookup = Dict(key(face.center) => index for (index, face) in pairs(mesh.faces))
    mixed = Set(data.mixed)
    largest = 0.0
    for index in data.mixed
        face = mesh.faces[index]
        window(face.center[1]) || continue
        slope = slope_of(face.center[1])
        normal = (-slope, 1.0) ./ hypot(slope, 1.0)
        tangential = 3 - face.axis
        centre = key(face.center)
        numerator = denominator = 0.0
        for dx in (-1, 1), dy in (-1, 1)
            candidate = get(lookup, (mod(centre[1] + dx, 2n), centre[2] + dy), 0)
            candidate in mixed || continue
            mesh.faces[candidate].axis == tangential || continue
            weight = data.liquid_fraction[candidate] * (1 - data.liquid_fraction[candidate])
            numerator += weight * jump[candidate]
            denominator += weight
        end
        interpolated = denominator > 0 ? numerator / denominator : 0.0
        largest = max(
            largest,
            abs(normal[face.axis] * jump[index] + normal[tangential] * interpolated),
        )
    end
    return largest
end

@testset "Linear jump inference: x and t impose the normal jump on all mixed faces" begin
    # A uniform jump J* = (1, 0) over tilted interfaces with equal densities:
    # the exact projection keeps only its tangential part. The constant
    # inference leaves the faces without a row untouched (at the slope 1 half of
    # the mixed faces), the normal rows clear their normal jump.
    sheet = (x, y, axis, phase) -> axis == 1 && phase == 2 ? 1.0 : 0.0
    for slope in (0.3, 1.0), n in (32, 64)
        graph = _triangle(0.5 + 0.37 / n, slope)
        residuals = map((ConstantJumpInference(), LinearJumpInference())) do inference
            result = _jump_inference_projection(
                n, (x, y) -> y - graph[1](x), sheet; inference,
            )
            @test result.stats.converged
            inference isa LinearJumpInference &&
                @test result.stats.jump_inference.normal_rows > 0
            _interpolated_normal_jump(result, graph[2]; window = _away_from_kinks)
        end
        if slope == 1
            # NUMERICS CONCERN (#171): at the slope 1 about half of the faces
            # without a regular row have |n_t| > |n_a| and get the β = 0 row
            # that infers their jump from the donors instead of the normal
            # row; for this uniform jump the interpolated normal jump stays
            # 0.225 (n = 32) and 0.255 (n = 64), against 0.48 for the
            # published rows and 0.057 / 0.099 when these faces keep their
            # jump. At the slope 0.3 the residual is 0.005. The β = 0 row
            # imposes the donors' inferred jump, not the normal-jump condition,
            # so neither freezing θ nor the dominance cap changes these values.
            @test_broken residuals[2] <= 0.01
            @test_broken residuals[2] <= 0.05 * residuals[1]
            @test residuals[2] <= 0.6 * residuals[1]
        else
            @test residuals[2] <= 0.01
            @test residuals[2] <= 0.05 * residuals[1]
        end
    end

    # On an interface between two grid lines every x-face of the band has an
    # exactly tangential normal: the constant inference gives none of them a
    # row, the extension gives each the β = 0 row, and the sheet is kept.
    n = 32
    graph = _wavy(0.5 + 0.37 / n, 0.0)
    for inference in (ConstantJumpInference(), LinearJumpInference())
        result = _jump_inference_projection(
            n, (x, y) -> y - graph[1](x), sheet; inference,
        )
        @test result.stats.converged
        @test result.stats.jump_inference.tangential_rows ==
            (inference isa LinearJumpInference ? n : 0)
        @test _largest_jump(result) ≈ 1 atol = 1.0e-10
        @test _interpolated_normal_jump(result, x -> 0.0) <= 1.0e-10
    end
end

_slope_weights(rows) = Dict(
    stencil.face => stencil.slope for stencil in rows if stencil.gas_extrapolation_face > 0
)
_frozen_slope_weights(workspace) =
    _slope_weights(workspace.pressure_cache.stencil_storage.resolved)

"""
Project the #171 idempotence fixture four times: a triangle at the slope 0.3,
the density ratio 10, the crossing-speed `limiter` and a gas predictor that
slides past the liquid by `slip`. With `frozen` the slope weights of the first
projection stay in the rows; without it every projection evaluates them from
the state it receives (what a history-free θ would do). Returns the largest
change of either phase velocity over repetitions 2–4, the first projection's
weights and the objects a later projection reuses.
"""
function _idempotence_fixture(; limiter, slip, frozen = true)
    n = 32
    graph = _triangle(0.5 + 0.37 / n, 0.3)
    mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic = (true, false))
    interface = InterfaceField(mesh, (x, y) -> y - graph[1](x); lipschitz = 1)
    inference = LinearJumpInference(; limiter)
    density = (10.0, 1.0)
    solver = CMDGFMProjectionSolver(interface; density, jump_inference = inference)
    workspace = CMDGFMProjectionWorkspace(solver)
    initial_velocity = () -> begin
        fields = (
            FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? sinpi(2y) : 0.3cospi(2x)),
            FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? slip + sinpi(2y) : 0.0),
        )
        foreach(phase -> fill!(fields[phase].boundary, 0.0), 1:2)
        fields
    end
    velocity = initial_velocity()
    data = CJI._projection_face_data(solver, workspace)
    stencils = filter(stencil -> stencil.resolved, CJI._cmdgfm_jump_stencils(solver, data))
    predictor = CJI._cmdgfm_limit_extrapolation!(
        copy(stencils), data, velocity, interface, inference,
    )
    tolerance = 1.0e-12
    changes = Float64[]
    for repetition in 1:4
        before = (copy(velocity[1].interior), copy(velocity[2].interior))
        frozen || CJI._cmdgfm_release_crossing!(workspace)
        stats = project!(
            velocity, CellField(mesh), solver, 1.0, workspace;
            tolerance, max_iterations = 4000,
        )
        @test stats.converged
        @test stats.jump_inference.normal_rows > 0
        # The weights of the first projection, evaluated from the predictor.
        frozen && @test _frozen_slope_weights(workspace) == _slope_weights(predictor)
        repetition == 1 && continue
        push!(
            changes,
            maximum(
                phase -> maximum(
                    index -> isfinite(before[phase][index]) ?
                        abs(velocity[phase].interior[index] - before[phase][index]) : 0.0,
                    eachindex(before[phase]),
                ),
                1:2,
            ),
        )
    end
    released_matches = released_differs = false
    if frozen
        # Released (as a flow does before each projection), the weights are
        # evaluated again, from the projected state.
        projected = CJI._cmdgfm_limit_extrapolation!(
            copy(stencils), data, velocity, interface, inference,
        )
        CJI._cmdgfm_release_crossing!(workspace)
        @test project!(
            velocity, CellField(mesh), solver, 1.0, workspace;
            tolerance, max_iterations = 4000,
        ).converged
        released_matches = _frozen_slope_weights(workspace) == _slope_weights(projected)
        released_differs = _frozen_slope_weights(workspace) != _slope_weights(predictor)
    end
    return (;
        mesh, interface, solver, workspace, data, stencils, predictor, initial_velocity,
        changes, released_matches, released_differs,
        fractional = count(stencil -> 0 < stencil.slope < 1, predictor),
        weighted = count(stencil -> stencil.gas_extrapolation_face > 0, predictor),
    )
end

@testset "Linear jump inference: the projection is idempotent" begin
    # Every row, including the normal rows of the faces that the constant
    # inference leaves without one, is a condition on the projected state,
    # and the slope weights θ of the crossing-speed limiter are frozen with the
    # rows of the reconstructed interface: projecting a projected state again
    # changes nothing. Both limiters run on the #171 fixture. The default
    # `:normal` fixture keeps its original slip 1 and has 7 of its 40 weighted
    # rows with θ in (0, 1); `:sliding` needs the gas 0.1 past the liquid,
    # because at slip 1 the sliding speed |u_g,x - u_l,x| |n_x| ≈ 0.29 puts
    # every row on the slope-0.3 interface at θ = 0 and no row would have a
    # weight to freeze, and then has 8 of the 40. Evaluated from the state each
    # projection receives instead (released), θ changes the largest phase
    # velocity by 0.33, 0.21, 0.062 (default) and 0.26, 0.11, 0.21 (sliding)
    # over repetitions 2–4, against no change at all when frozen.
    cases = ((:normal, 1.0), (:sliding, 0.1))
    frozen = map(cases) do (limiter, slip)
        fixture = _idempotence_fixture(; limiter, slip, frozen = true)
        println(
            "idempotence limiter=$limiter ",
            "fractional=$(fixture.fractional)/$(fixture.weighted) ",
            "frozen changes=$(fixture.changes)",
        )
        @test fixture.fractional > 0
        @test all(fixture.changes .<= 1.0e-10)
        @test fixture.released_matches
        @test fixture.released_differs
        fixture
    end
    for (limiter, slip) in cases
        released = _idempotence_fixture(; limiter, slip, frozen = false)
        println("idempotence limiter=$limiter released changes=$(released.changes)")
        # Measured 0.33/0.21/0.062 (default) and 0.26/0.11/0.21 (sliding); this
        # bound separates them from the frozen change by a factor of at least 6.
        @test all(released.changes .> 1.0e-2)
    end

    # Reusing the same workspace and reconstruction with another limiter
    # must not inherit the previous limiter's frozen coefficients.
    sliding = last(frozen)
    normal = CMDGFMProjectionSolver(sliding.interface; density = (10.0, 1.0))
    normal_rows = CJI._cmdgfm_limit_extrapolation!(
        copy(sliding.stencils), sliding.data, sliding.initial_velocity(),
        sliding.interface, normal.jump_inference,
    )
    @test _slope_weights(normal_rows) != _slope_weights(sliding.predictor)
    for (policy, rows) in ((normal, normal_rows), (sliding.solver, sliding.predictor))
        @test project!(
            sliding.initial_velocity(), CellField(sliding.mesh), policy, 1.0,
            sliding.workspace;
            tolerance = 1.0e-12, max_iterations = 4000, warm_start = false,
        ).converged
        @test _frozen_slope_weights(sliding.workspace) == _slope_weights(rows)
    end
end

@testset "Linear jump inference: a resolved shear layer stays continuous in a flow" begin
    # Issue #171 reproducer: both phases carry the same resolved shear layer
    # u = tanh(y/δ)/2 (δ/h = 3.2) across y = 0.01cos(2πx), equal densities and
    # viscosities, no surface tension. The exact solution keeps the velocity
    # continuous. With the thesis rows the first projection creates a
    # tangential phase-velocity jump of about 2U'h (0.30); the linear
    # inference leaves 0.05 after ten steps.
    n, delta = 64, 0.05
    largest = map((ConstantJumpInference(), LinearJumpInference())) do inference
        mesh = BlockTreeMesh(;
            base_mesh = (n, n), block_size = (8, 8), origin = (0.0, -0.5),
            periodic = (true, false),
        )
        interface = InterfaceField(mesh, (x, y) -> y - 0.01cospi(2x); lipschitz = 1)
        flow = FlowSimulation(
            mesh, TwoPhase(interface; density = (1.0, 1.0), dynamic_viscosity = (1.0e-3, 1.0e-3));
            velocity_model = TwoVelocityModel(
                pressure_projection = CMDGFMProjection(jump_inference = inference),
            ),
            velocity = (x, y, time, axis) -> axis == 1 ? 0.5tanh(y / delta) : 0.0,
            boundary = FreeSlip(),
        )
        dt = stable_timestep(flow; cfl = 0.4)
        for _ in 1:10
            step!(flow, dt)
        end
        liquid, gas = flow.velocity
        jump = 0.0
        for (index, face) in pairs(mesh.faces)
            face.axis == 1 || continue
            value = gas.interior[index] - liquid.interior[index]
            isfinite(value) && (jump = max(jump, abs(value)))
        end
        jump
    end
    @test largest[1] >= 0.2
    @test largest[2] <= 0.08
    @test largest[2] <= 0.25 * largest[1]
end
