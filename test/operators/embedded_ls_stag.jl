using SparseArrays: spdiagm
using LinearAlgebra: Diagonal, Symmetric, diag, eigmin

# Fixtures of the #110 convergence tests in `embedded_wall_rows.jl`: domain
# (0, 8)^2, the right half refined once on the composite mesh, manufactured
# fields that vanish on the solid and on the domain boundary.
const LS_STAG_EXTENT = 8.0

ls_stag_mesh(n, refined) = BlockTreeMesh(;
    base_mesh = (n, n), block_size = (4, 4), extent = (LS_STAG_EXTENT, LS_STAG_EXTENT),
    periodic = (false, false), max_level = 1,
    refine = (key, origin, extent) -> refined && key.level == 0 && key.i >= div(n, 8),
)

function ls_stag_slope_solution(slope, x, y)
    L = LS_STAG_EXTENT
    wall = 1 + slope * x
    a = pi * x / L
    b = pi * (y - wall) / (L - wall)
    ax = pi / L
    bx = pi * slope * (y - L) / (L - wall)^2
    by = pi / (L - wall)
    bxx = 2pi * slope^2 * (y - L) / (L - wall)^3
    value = sin(a) * sin(b)
    laplacian = -(ax^2 + bx^2 + by^2) * value +
        2ax * cos(a) * cos(b) * bx + sin(a) * cos(b) * bxx
    return value, laplacian
end

# Fluid outside the circle of radius 1.3 around (4, 4).
function ls_stag_circle_solution(x, y)
    a = pi / LS_STAG_EXTENT
    radius = 1.3
    s = sin(a * x) * sin(a * y)
    dx, dy = x - 4, y - 4
    q = (dx^2 + dy^2 - radius^2) / radius^2
    laplacian = -2a^2 * s * q + 4s / radius^2 +
        2 * (a * cos(a * x) * sin(a * y) * 2dx + a * sin(a * x) * cos(a * y) * 2dy) /
        radius^2
    return s * q, laplacian
end

function ls_stag_case(kind)
    kind === :circle && return (;
        levelset = (x, y) -> hypot(x - 4, y - 4) - 1.3, lipschitz = 1.0,
        solution = ls_stag_circle_solution,
    )
    slope = kind === :slope01 ? 0.1 : 0.5
    return (;
        levelset = (x, y) -> y - 1 - slope * x, lipschitz = hypot(1, slope),
        solution = (x, y) -> ls_stag_slope_solution(slope, x, y), slope,
    )
end

"""Where `LSStagClosure` places each value: the midpoint of the open part of
its face (the face centre for a closed face), and which faces are open."""
function ls_stag_value_points(geometry, component)
    mesh = geometry.mesh
    points = NTuple{2, Float64}[]
    open = Bool[]
    for global_index in component.local_to_global
        face = mesh.faces[global_index]
        aperture = geometry.aperture[global_index]
        is_open = aperture > 1.0e-10 && geometry.face_dual_volumes[global_index] > 0
        push!(open, is_open)
        if is_open
            _, _, lower, upper, _, _ = TwoPhaseAMR._ls_stag_open_part(geometry, face, aperture)
            middle = (lower + upper) / 2
            push!(points, face.axis == 1 ? (face.center[1], middle) : (middle, face.center[2]))
        else
            push!(points, face.center)
        end
    end
    return points, open
end

@testset "LS-STAG closure is second order at oblique walls (#110)" begin
    # The Helmholtz problem of the #110 tests, solved with `LSStagClosure`
    # and measured at the value points, over every face with a positive open
    # aperture (faces whose centre lies in the solid included).
    function errors(kind, n, refined)
        case = ls_stag_case(kind)
        geometry = GeometryField(
            ls_stag_mesh(n, refined), case.levelset; lipschitz = case.lipschitz,
        )
        operator = StaggeredDiffusionOperator(
            geometry; viscosity = 1.0, closure = LSStagClosure(),
        )
        return map((operator.x, operator.y)) do component
            points, open = ls_stag_value_points(geometry, component)
            values = [case.solution(point...) for point in points]
            exact = [open[i] ? values[i][1] : 0.0 for i in eachindex(points)]
            rhs = [open[i] ? values[i][1] - values[i][2] : 0.0 for i in eachindex(points)]
            numerical = (spdiagm(0 => ones(length(rhs))) - component.laplacian) \ rhs
            error = (numerical - exact)[open]
            volume = geometry.face_dual_volumes[component.local_to_global][open]
            (
                maximum(abs, error),
                sqrt(sum(volume .* error .^ 2) / sum(volume)),
                sum(volume .* abs.(error)) / sum(volume),
            )
        end
    end
    # (max, L2, L1) of components 1 and 2 at n = 64, as measured; the default
    # closure gives 0.0435, 0.0644 (max) and 0.0023, 0.0029 (L2) on the
    # uniform slope-0.1 mesh. Orders 16 -> 32 -> 64 are 1.98 to 2.07 in L2
    # and L1 and 1.68 to 2.00 in the maximum norm, except for component 2 at
    # slope 0.5 (below).
    expected = (
        (:slope01, false) => ((7.163e-5, 3.658e-5, 3.039e-5), (7.139e-5, 3.685e-5, 3.111e-5)),
        (:slope01, true) => ((9.869e-5, 2.492e-5, 1.88e-5), (2.509e-4, 6.239e-5, 3.969e-5)),
        (:slope05, false) => ((2.172e-4, 1.009e-4, 8.2e-5), (5.385e-4, 1.134e-4, 9.044e-5)),
        (:slope05, true) => ((2.298e-4, 5.645e-5, 4.487e-5), (3.794e-4, 9.851e-5, 6.792e-5)),
        (:circle, false) => ((6.787e-3, 1.818e-3, 1.135e-3), (6.787e-3, 1.818e-3, 1.135e-3)),
        (:circle, true) => ((6.461e-3, 1.25e-3, 6.975e-4), (6.787e-3, 1.337e-3, 7.563e-4)),
    )
    for ((kind, refined), reference) in expected
        history = [errors(kind, n, refined) for n in (16, 32, 64)]
        for axis in 1:2, norm in 1:3
            series = [entry[axis][norm] for entry in history]
            @test series[end] <= 1.02 * reference[axis][norm]
            # The maximum error of component 2 at slope 0.5 on the uniform mesh
            # sits in the corner where the wall meets the boundary x = 8 under
            # an interior angle of 116.6 degrees (pi/angle = 1.54): orders 1.31
            # and 1.54, 1.77 for 64 -> 128.
            gate = norm == 1 ? (kind === :slope05 && axis == 2 ? 1.25 : 1.6) : 1.85
            for level in 1:2
                @test log2(series[level] / series[level + 1]) >= gate
            end
        end
    end
end

@testset "LS-STAG closure keeps the unknowns, the mass and an SPD stiffness (#110)" begin
    solid_centred = 0
    for kind in (:slope01, :circle), refined in (false, true)
        case = ls_stag_case(kind)
        n = 16
        h = LS_STAG_EXTENT / n
        mesh = ls_stag_mesh(n, refined)
        geometry = GeometryField(mesh, case.levelset; lipschitz = case.lipschitz)
        default = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
        lsstag = StaggeredDiffusionOperator(
            geometry; viscosity = 1.0, closure = LSStagClosure(),
        )
        for (current, component) in ((default.x, lsstag.x), (default.y, lsstag.y))
            stiffness = component.stiffness
            @test stiffness == transpose(stiffness)
            scale = Diagonal(1 ./ sqrt.(diag(stiffness)))
            @test eigmin(Symmetric(Matrix(scale * stiffness * scale))) > 1.0e-3
            # Same unknowns and the same mass, the open staggered volumes.
            @test component.local_to_global == current.local_to_global
            @test component.volumes == current.volumes
            dual = geometry.face_dual_volumes[component.local_to_global]
            @test component.volumes[dual .> 0] == dual[dual .> 0]
            solid_centred += count(
                global_index -> geometry.aperture[global_index] > 0 &&
                    case.levelset(mesh.faces[global_index].center...) < 0,
                component.local_to_global,
            )
            # Rows away from the solid keep the default closure.
            far = [
                local_index for (local_index, global_index) in pairs(component.local_to_global)
                    if abs(case.levelset(mesh.faces[global_index].center...)) > 4h
            ]
            @test !isempty(far)
            @test stiffness[far, :] == current.stiffness[far, :]
            @test component.laplacian[far, :] == current.laplacian[far, :]
        end
    end
    # Faces with a positive aperture whose centre lies in the solid are
    # ordinary unknowns.
    @test solid_centred > 0
end

@testset "LS-STAG rows of an affine field that vanishes on the wall are exact (#110)" begin
    for kind in (:slope01, :slope05), refined in (false, true), n in (16, 32)
        case = ls_stag_case(kind)
        h = LS_STAG_EXTENT / n
        geometry = GeometryField(
            ls_stag_mesh(n, refined), case.levelset; lipschitz = case.lipschitz,
        )
        operator = StaggeredDiffusionOperator(
            geometry; viscosity = 1.0, closure = LSStagClosure(),
        )
        for component in (operator.x, operator.y)
            points, open = ls_stag_value_points(geometry, component)
            distance = [
                open[i] ? case.levelset(points[i]...) / case.lipschitz : 0.0
                    for i in eachindex(points)
            ]
            rows = [
                i for i in eachindex(points)
                    if open[i] && distance[i] < 2h &&
                    2h < points[i][1] < LS_STAG_EXTENT - 2h
            ]
            @test !isempty(rows)
            residual = component.stiffness * distance
            magnitude = abs.(component.stiffness) * abs.(distance)
            @test maximum(abs.(residual[rows]) ./ magnitude[rows]) <= 1.0e-10
        end
    end
end

@testset "LS-STAG viscous multigrid converges like the default closure (#110)" begin
    # No distance floor: a small cut cell has the large conductance of its
    # width. The smoother's damping depends on the ratio of a row's absolute
    # sum to its diagonal, not on its size, so the V-cycle copes. A cut circle
    # on the composite mesh, and the radius among 64 offsets within one fine
    # width that leaves the smallest open cut cell (open fraction 6.8e-6).
    # Measured cycles (largest component) at reaction 1, diffusion 1:
    # 21 and 20, and 22 and 26 (default, LS-STAG); over n = 32 to 256 and the
    # worst offsets of a circle and a slope the LS-STAG count stays within
    # 1.74 times the default one, and below it where the default closure's
    # own small rows lower the smoother weight (0.2 against 0.68).
    fine_width = LS_STAG_EXTENT / 64
    for radius in (1.3, 1.3 + 58 * fine_width / 64 + fine_width / 448)
        mesh = ls_stag_mesh(32, true)
        geometry = GeometryField(mesh, (x, y) -> hypot(x - 4, y - 4) - radius; lipschitz = 1)
        rhs = FaceVelocity(mesh, (x, y, time, axis) -> sin(x + 2axis) * cos(y - axis))
        cycles = map((WallDistanceClosure(), LSStagClosure())) do closure
            operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0, closure)
            solver = StaggeredEllipticSolver(
                StaggeredEllipticOperator(operator; reaction = 1, diffusion = 1),
            )
            solution = FaceVelocity(mesh)
            stats = solve!(solution, solver, rhs; tolerance = 1.0e-10, max_cycles = 200)
            @test stats.converged
            max(stats.x.cycles, stats.y.cycles)
        end
        @test cycles[2] <= 1.5 * cycles[1]
    end
end
