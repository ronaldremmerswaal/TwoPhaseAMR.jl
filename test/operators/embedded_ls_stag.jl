using SparseArrays: spdiagm
using LinearAlgebra: Diagonal, Symmetric, diag, dot, eigmin

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
        return map((1, 2)) do axis
            component = axis == 1 ? operator.x : operator.y
            nodes = value_points(operator, axis)
            points, open = nodes.points, nodes.volumes .> 0
            values = [case.solution(point...) for point in points]
            exact = [open[i] ? values[i][1] : 0.0 for i in eachindex(points)]
            rhs = [open[i] ? values[i][1] - values[i][2] : 0.0 for i in eachindex(points)]
            numerical = (spdiagm(0 => ones(length(rhs))) - component.laplacian) \ rhs
            error = (numerical - exact)[open]
            volume = nodes.volumes[open]
            (
                maximum(abs, error),
                sqrt(sum(volume .* error .^ 2) / sum(volume)),
                sum(volume .* abs.(error)) / sum(volume),
            )
        end
    end
    # (max, L2, L1) of components 1 and 2 at n = 64, as measured; the
    # wall-distance closure gives 0.0435, 0.0644 (max) and 0.0023, 0.0029 (L2)
    # on the uniform slope-0.1 mesh. Orders 16 -> 32 -> 64 are 1.98 to 2.07 in L2
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
        # This test compares the LS-STAG rows with the legacy closure; name
        # both explicitly so the reference does not depend on the default.
        legacy = StaggeredDiffusionOperator(
            geometry; viscosity = 1.0, closure = WallDistanceClosure(),
        )
        lsstag = StaggeredDiffusionOperator(
            geometry; viscosity = 1.0, closure = LSStagClosure(),
        )
        for (current, component) in ((legacy.x, lsstag.x), (legacy.y, lsstag.y))
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
            # Rows away from the solid agree with the legacy closure.
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
        for (axis, component) in ((1, operator.x), (2, operator.y))
            nodes = value_points(operator, axis)
            points, open = nodes.points, nodes.volumes .> 0
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

@testset "value_points of the staggered diffusion operator" begin
    for T in (Float32, Float64)
        # The #110 circle (fluid outside) on a composite mesh.
        mesh = BlockTreeMesh(;
            base_mesh = (16, 16), block_size = (4, 4), origin = (zero(T), zero(T)),
            extent = (T(LS_STAG_EXTENT), T(LS_STAG_EXTENT)), periodic = (false, false),
            max_level = 1, refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
        )
        levelset = (x, y) -> hypot(x - 4, y - 4) - T(1.3)
        geometry = GeometryField(mesh, levelset; lipschitz = one(T))
        lsstag = StaggeredDiffusionOperator(geometry; closure = LSStagClosure())
        wall = StaggeredDiffusionOperator(geometry; closure = WallDistanceClosure())
        moved_into_fluid = 0
        for axis in 1:2
            nodes = value_points(lsstag, axis)
            legacy = value_points(wall, axis)
            component = axis == 1 ? lsstag.x : lsstag.y
            @test nodes.faces == component.local_to_global == legacy.faces
            @test nodes.faces !== component.local_to_global
            @test eltype(nodes.points) == NTuple{2, T}
            @test eltype(nodes.volumes) == T
            centres = [T.(mesh.faces[index].center) for index in nodes.faces]
            areas = [T(mesh.faces[index].area) for index in nodes.faces]
            apertures = geometry.aperture[nodes.faces]
            open_volumes = geometry.face_dual_volumes[nodes.faces]
            fluid = findall((apertures .> T(1.0e-10)) .& (open_volumes .> 0))
            others = setdiff(eachindex(nodes.faces), fluid)
            # Wall distance: every value at its face centre.
            @test legacy.points == centres
            @test legacy.volumes == ifelse.(apertures .> 0, open_volumes, zero(T))
            # LS-STAG: a fluid value carries the open volume of its face; any
            # other value (covered faces among them) sits at its centre.
            @test nodes.volumes[fluid] == open_volumes[fluid]
            @test all(iszero, nodes.volumes[others])
            @test nodes.points[others] == centres[others]
            @test any(index -> iszero(apertures[index]), others)
            # A fluid value lies on its face, at the midpoint of the open part,
            # which touches one end of the face (a nearly full face, within
            # 256 eps, is taken as full).
            tangent = 3 - axis
            @test all(index -> nodes.points[index][axis] == centres[index][axis], fluid)
            offset = map(fluid) do index
                lower = centres[index][tangent] - areas[index] / 2
                upper = centres[index][tangent] + areas[index] / 2
                half = apertures[index] * areas[index] / 2
                position = nodes.points[index][tangent]
                min(abs(position - lower - half), abs(upper - half - position)) / areas[index]
            end
            @test maximum(offset) <= 256eps(T)
            # In the fluid: the PLIC wall deviates from the circle by about a
            # tenth of a cell here, so the midpoint of an open part of at least
            # a quarter of the face lies outside it.
            @test all(index -> apertures[index] < T(0.25) || levelset(nodes.points[index]...) > 0, fluid)
            moved_into_fluid += count(
                index -> levelset(centres[index]...) < 0 < levelset(nodes.points[index]...), fluid,
            )
        end
        # Cut faces whose centre lies in the solid carry fluid values.
        @test moved_into_fluid > 0
        @test_throws ArgumentError value_points(lsstag, 3)
        # Without a solid every value sits at its face centre.
        plain = StaggeredDiffusionOperator(mesh)
        nodes = value_points(plain, 2)
        @test nodes.points == [T.(mesh.faces[index].center) for index in nodes.faces]
        @test nodes.volumes == plain.y.volumes
    end
end

@testset "LS-STAG viscous multigrid versus wall distance (#110)" begin
    # No distance floor: a small cut cell has the large conductance of its
    # width. The smoother's damping depends on the ratio of a row's absolute
    # sum to its diagonal, not on its size, so the V-cycle copes. A cut circle
    # on the composite mesh, and the radius among 64 offsets within one fine
    # width that leaves the smallest open cut cell (open fraction 6.8e-6).
    # Measured cycles (largest component) at reaction 1, diffusion 1:
    # 21 and 20, and 22 and 26 (wall distance, LS-STAG); over n = 32 to 256
    # and the worst offsets of a circle and a slope the LS-STAG count stays
    # within 1.74 times the wall-distance count, and below it where small
    # wall-distance rows lower the smoother weight (0.2 against 0.68).
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

@testset "LS-STAG keeps moving domain walls off a stationary solid" begin
    for T in (Float32, Float64)
        mesh = BlockTreeMesh(
            block_size = (16, 16), periodic = (false, false),
            origin = (zero(T), zero(T)), extent = (one(T), one(T)),
        )
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - T(0.5), y - T(0.5)) - T(0.18);
            lipschitz = one(T),
        )
        time = T(0.375)
        moving = NoSlip(
            (x, y, t, axis) -> (axis == 1 ? one(T) : -T(2)) * (one(T) + t),
        )
        sides = VelocityBoundaryConditions(
            left = moving, right = moving, bottom = moving, top = moving,
        )
        operator = StaggeredDiffusionOperator(
            geometry; viscosity = one(T), boundary = moving,
            closure = LSStagClosure(),
        )
        side_operator = StaggeredDiffusionOperator(
            geometry; viscosity = one(T), boundary = sides,
            closure = LSStagClosure(),
        )
        @test operator.closure isa LSStagClosure
        velocity = FaceVelocity(mesh)
        tendency = FaceVelocity(mesh)
        side_tendency = FaceVelocity(mesh)
        diffusion!(tendency, velocity, operator; time)
        diffusion!(side_tendency, velocity, side_operator; time)
        @test tendency.interior == side_tendency.interior
        @test all(iszero, tendency.boundary)
        wall_source = zeros(T, length(mesh.faces))
        for axis in 1:2
            component = axis == 1 ? operator.x : operator.y
            side_component = axis == 1 ? side_operator.x : side_operator.y
            @test component.stiffness == side_component.stiffness
            solid = filter(
                c -> iszero(c.boundary_axis), component.wall_connections,
            )
            domain = filter(
                c -> !iszero(c.boundary_axis) && c.tangential,
                component.wall_connections,
            )
            @test any(c -> c.tangential, solid)
            @test !isempty(domain)
            # The moving callback is nonzero on the domain walls but must not
            # reach the stationary embedded solid.
            @test all(
                c -> iszero(TwoPhaseAMR._wall_value(moving, c, time, axis, T)),
                solid,
            )
            solid_indices = [
                component.local_to_global[c.degree_of_freedom] for c in solid
            ]
            domain_indices = [
                component.local_to_global[c.degree_of_freedom] for c in domain
            ]
            @test isempty(intersect(solid_indices, domain_indices))
            @test all(i -> iszero(tendency.interior[i]), solid_indices)
            @test any(i -> !iszero(tendency.interior[i]), domain_indices)
            TwoPhaseAMR._add_wall_source!(
                wall_source, component, operator, one(T), time, axis,
            )
        end
        @test wall_source == tendency.interior
    end
end

@testset "LS-STAG slip conditions at an embedded wall" begin
    mesh = BlockTreeMesh(
        block_size = (16, 16), periodic = (true, false),
        origin = (0.0, 0.0), extent = (1.0, 1.0),
    )
    wall = 4.0 / 16.0
    geometry = GeometryField(mesh, (x, y) -> y - wall; lipschitz = 1.0)
    no_slip = StaggeredDiffusionOperator(
        geometry; viscosity = 1.0, boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    free_slip = StaggeredDiffusionOperator(
        geometry; viscosity = 1.0, boundary = FreeSlip(),
        closure = LSStagClosure(),
    )
    @test no_slip.closure isa LSStagClosure
    @test free_slip.closure isa LSStagClosure
    @test no_slip.x.local_to_global == free_slip.x.local_to_global
    @test no_slip.x.volumes == free_slip.x.volumes
    @test no_slip.y.local_to_global == free_slip.y.local_to_global
    @test no_slip.y.volumes == free_slip.y.volumes
    for component in (no_slip.x, free_slip.x, no_slip.y, free_slip.y)
        @test component.stiffness == transpose(component.stiffness)
    end

    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0,
    )
    for index in eachindex(velocity.interior)
        geometry.face_dual_volumes[index] > 0 ||
            (velocity.interior[index] = 0.0)
    end
    velocity.interior[no_slip.x.dirichlet_faces] .= 0.0
    velocity.interior[no_slip.y.dirichlet_faces] .= 0.0
    xlocal = no_slip.x.local_to_global
    ux = velocity.interior[xlocal]

    no_tendency = FaceVelocity(mesh)
    free_tendency = FaceVelocity(mesh)
    diffusion!(no_tendency, velocity, no_slip)
    diffusion!(free_tendency, velocity, free_slip)
    # A constant field tangential to a free-slip wall has no strain at all.
    @test maximum(abs, free_tendency.interior[xlocal]) <= 256eps()
    @test any(!iszero, no_tendency.interior[xlocal])

    difference = no_slip.x.stiffness - free_slip.x.stiffness
    @test difference == transpose(difference)
    @test eigmin(Symmetric(Matrix(difference))) >=
        -1024eps() * maximum(abs, difference)
    @test dot(ux, no_slip.x.stiffness * ux) > 0
    @test dot(ux, free_slip.x.stiffness * ux) <= 256eps()
    @test dot(ux, difference * ux) > 0

    h = 1.0 / 16.0
    drag_face = findfirst(
        face -> face.axis == 1 &&
            isapprox(face.center[2], wall + h / 2; atol = 64eps()),
        mesh.faces,
    )
    @test !isnothing(drag_face)
    @test no_tendency.interior[drag_face] < 0
end

@testset "Per-side walls carry their own embedded-solid condition" begin
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        periodic = (false, false),
        refine = (key, origin, _) -> key.level == 0 && origin[1] >= 0.5,
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.25 - 0.5x; lipschitz = hypot(1.0, 0.5))
    @test VelocityBoundaryConditions().embedded isa NoSlip
    for embedded in (NoSlip((x, y, time, axis) -> 1.0), Outflow())
        @test_throws ArgumentError VelocityBoundaryConditions(; embedded)
    end
    sides = (left = FreeSlip(), right = NoSlip(), bottom = NoSlip(), top = FreeSlip())
    free = (left = FreeSlip(), right = FreeSlip(), bottom = FreeSlip(), top = FreeSlip())
    for closure in (LSStagClosure(), WallDistanceClosure())
        function stiffness(boundary)
            operator = StaggeredDiffusionOperator(geometry; viscosity = 0.3, boundary, closure)
            return (operator.x.stiffness, operator.y.stiffness)
        end
        # The default solid is a stationary no-slip wall, as before; a solid
        # and sides that all match a uniform condition reproduce it bitwise.
        no_slip_solid = stiffness(VelocityBoundaryConditions(; sides...))
        @test no_slip_solid == stiffness(VelocityBoundaryConditions(; sides..., embedded = NoSlip()))
        @test stiffness(VelocityBoundaryConditions()) == stiffness(NoSlip())
        @test stiffness(VelocityBoundaryConditions(; free..., embedded = FreeSlip())) ==
            stiffness(FreeSlip())
        # Releasing the solid at fixed sides removes dissipation only.
        free_solid = stiffness(VelocityBoundaryConditions(; sides..., embedded = FreeSlip()))
        for (no, released) in zip(no_slip_solid, free_solid)
            loss = no - released
            @test any(!iszero, loss)
            @test eigmin(Symmetric(Matrix(loss))) >= -1024eps() * maximum(abs, no)
        end
    end
    # A constant velocity along a free-slip solid has no strain there under a
    # no-slip lid; a no-slip solid damps it.
    channel = BlockTreeMesh(
        block_size = (16, 16), periodic = (true, false),
        origin = (0.0, 0.0), extent = (1.0, 1.0),
    )
    floor = GeometryField(channel, (x, y) -> y - 0.25; lipschitz = 1.0)
    velocity = FaceVelocity(channel, (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0)
    for index in eachindex(velocity.interior)
        floor.face_dual_volumes[index] > 0 || (velocity.interior[index] = 0.0)
    end
    near_floor = [
        index for (index, face) in pairs(channel.faces) if face.axis == 1 && face.center[2] < 0.5
    ]
    for embedded in (FreeSlip(), NoSlip())
        operator = StaggeredDiffusionOperator(
            floor; viscosity = 1.0, closure = LSStagClosure(),
            boundary = VelocityBoundaryConditions(; top = NoSlip(), embedded),
        )
        velocity.interior[operator.x.dirichlet_faces] .= 0.0
        tendency = FaceVelocity(channel)
        diffusion!(tendency, velocity, operator)
        damping = maximum(index -> abs(tendency.interior[index]), near_floor)
        @test embedded isa FreeSlip ? damping <= 256eps() : damping > 1.0e-3
        @test maximum(abs, tendency.interior) > 1.0e-3
    end
end

@testset "LS-STAG wall traction of a linear shear" begin
    mesh = BlockTreeMesh(
        block_size = (16, 16), periodic = (true, false),
        origin = (0.0, 0.0), extent = (1.0, 1.0),
    )
    wall = 4.0 / 16.0
    geometry = GeometryField(mesh, (x, y) -> y - wall; lipschitz = 1.0)
    viscosity = 0.3
    operator = StaggeredDiffusionOperator(
        geometry; viscosity, boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test operator.closure isa LSStagClosure
    # A piecewise-linear profile that vanishes at the wall and is flat zero
    # well before the opposite (no-slip) boundary: only the embedded wall then
    # contributes to the total traction, and the kink cancels in the sum.
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            max(0.0, min(y - wall, 0.5 - (y - wall))) : 0.0,
    )
    for index in eachindex(velocity.interior)
        geometry.face_dual_volumes[index] > 0 ||
            (velocity.interior[index] = 0.0)
    end
    velocity.interior[operator.x.dirichlet_faces] .= 0.0
    velocity.interior[operator.y.dirichlet_faces] .= 0.0
    local_velocity = velocity.interior[operator.x.local_to_global]
    # The component stiffness is geometric; `diffusion!` applies the
    # viscosity. Include it to compare with the physical traction.
    force = operator.viscosity * (operator.x.stiffness * local_velocity)
    # The integrated wall traction of the unit-length wall is mu * 1. The
    # sign depends on the side convention, so pin its magnitude.
    @test isapprox(abs(sum(force)), viscosity; rtol = 1.0e-8, atol = 1.0e-10)

    tendency = FaceVelocity(mesh)
    diffusion!(tendency, velocity, operator)
    h = 1.0 / 16.0
    first_face = findfirst(
        face -> face.axis == 1 &&
            isapprox(face.center[2], wall + h / 2; atol = 64eps()),
        mesh.faces,
    )
    @test !isnothing(first_face)
    # The profile is affine between the wall and this first fluid face, so the
    # conservative wall flux and the interior flux cancel there. The nonzero
    # wall stress is asserted by the integrated traction above.
    @test abs(tendency.interior[first_face]) <= 1.0e-10
end
