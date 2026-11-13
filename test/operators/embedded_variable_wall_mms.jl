module EmbeddedVariableWallMMS

using Test, TwoPhaseAMR
using LinearAlgebra: mul!, norm
using SparseArrays: nonzeros, spdiagm

const L = 8.0
const K = 2pi / L
const RESOLUTIONS = (16, 32, 64)

# Y = (1-z^2)^2(1+a*z^2), expanded so its first three derivatives are
# evaluated independently of any discrete differentiation in the operator.
function profile(z, a)
    y = 1 + (a - 2) * z^2 + (1 - 2a) * z^4 + a * z^6
    dy = 2(a - 2) * z + 4(1 - 2a) * z^3 + 6a * z^5
    ddy = 2(a - 2) + 12(1 - 2a) * z^2 + 30a * z^4
    dddy = 24(1 - 2a) * z + 120a * z^3
    return y, dy, ddy, dddy
end

function layer_value_laplacian(x, y, axis, middle, halfheight, a)
    z = (y - middle) / halfheight
    Y, Yz, Yzz, Yzzz = profile(z, a)
    if axis == 1
        value = sin(K * x) * Yz / halfheight
        laplacian = sin(K * x) * (Yzzz / halfheight^3 - K^2 * Yz / halfheight)
    else
        value = -K * cos(K * x) * Y
        laplacian = -K * cos(K * x) * (Yzz / halfheight^2 - K^2 * Y)
    end
    return value, laplacian
end

function exact_value_forcing(x, y, axis, middle, halfheight, viscosities)
    low, high = viscosities
    a_low = 2.0
    a_high = 2 + (halfheight^2 * K^2 / 2) * (low / high - 1)
    if y == middle
        # A normal-velocity face here owns equal portions of the two layers.
        # The one-sided forces differ although the velocity and traction do not.
        value_low, lap_low = layer_value_laplacian(
            x, y, axis, middle, halfheight, a_low,
        )
        value_high, lap_high = layer_value_laplacian(
            x, y, axis, middle, halfheight, a_high,
        )
        value = (value_low + value_high) / 2
        return value, value - (low * lap_low + high * lap_high) / 2
    end
    viscosity, a = y < middle ? (low, a_low) : (high, a_high)
    value, laplacian = layer_value_laplacian(x, y, axis, middle, halfheight, a)
    return value, value - viscosity * laplacian
end

function fixture(n, refined, embedded, viscosities; match_embedded_domain = false)
    matched_open = !embedded && match_embedded_domain
    mesh = BlockTreeMesh(;
        base_mesh = matched_open ? (n, 7n ÷ 8) : (n, n),
        block_size = matched_open ? (2, 2) : (4, 4),
        origin = matched_open ? (0.0, 1.0) : (0.0, 0.0),
        extent = matched_open ? (L, L - 1) : (L, L),
        periodic = (true, false), max_level = 1,
        refine = (key, origin, extent) ->
        refined && key.level == 0 && origin[1] >= L / 2,
    )
    geometry = if embedded
        solid = InterfaceField(mesh)
        initialize!(solid, (0.0, 1.0), 1.0)
        GeometryField(solid)
    else
        nothing
    end
    middle, halfheight = embedded || matched_open ? (4.5, 3.5) : (4.0, 4.0)
    interface = InterfaceField(mesh; geometry)
    initialize!(interface, (0.0, 1.0), middle)
    operator = StaggeredDiffusionOperator(
        interface; viscosity = viscosities, boundary = NoSlip(),
    )
    @test operator isa TwoPhaseAMR.VariableStaggeredDiffusionOperator
    volumes = embedded ? geometry.face_dual_volumes : mesh.face_dual_volumes

    # The liquid is the lower, first-viscosity layer; verify that this really
    # is a phase interface, including its owned horizontal faces on both grids.
    interface_faces = [
        face for (i, face) in pairs(mesh.faces)
            if face.axis == 2 && face.center[2] == middle && volumes[i] > 0
    ]
    @test !isempty(interface_faces)
    @test all(interface_faces) do face
        cm = TwoPhaseAMR._conditional_fraction(interface, face.minus)
        cp = TwoPhaseAMR._conditional_fraction(interface, face.plus)
        (cm == 1 && cp == 0) || (cm == 0 && cp == 1)
    end
    return (; mesh, geometry, interface, operator, volumes, middle, halfheight)
end

function component_norm(mesh, volumes, values, axis)
    indices = [
        i for (i, face) in pairs(mesh.faces)
            if face.axis == axis && volumes[i] > 0
    ]
    @test !isempty(indices)
    # Include every physical staggered dual volume, including the prescribed
    # wall-normal face; only genuinely covered, zero-volume entries are absent.
    return sqrt(
        sum(volumes[i] * abs2(values[i]) for i in indices) /
            sum(volumes[i] for i in indices)
    )
end

function helmholtz_errors(n, refined, embedded, viscosities)
    data = fixture(n, refined, embedded, viscosities)
    (; mesh, geometry, operator, volumes, middle, halfheight) = data
    exact = zeros(length(mesh.faces))
    rhs = similar(exact)
    constrained = falses(length(exact))
    constrained[operator.dirichlet_faces] .= true
    for (i, face) in pairs(mesh.faces)
        if volumes[i] > 0 && !constrained[i]
            value, forcing = exact_value_forcing(
                face.center..., face.axis, middle, halfheight, viscosities,
            )
            exact[i] = value
            rhs[i] = operator.volumes[i] * forcing
        else
            rhs[i] = 0
        end
    end
    # Covered entries have algebraic extensions, not physical dual volumes.
    # A prescribed wall face has zero RHS even if its geometric volume is > 0.
    matrix = spdiagm(0 => operator.volumes) + operator.stiffness
    numerical = matrix \ rhs
    error = numerical - exact
    errors = ntuple(axis -> component_norm(mesh, volumes, error, axis), 2)
    for axis in 1:2
        @test component_norm(mesh, volumes, exact, axis) > 0.05
    end
    @test all(isfinite, numerical)
    @test all(iszero, numerical[operator.dirichlet_faces])

    return (; errors)
end

function operator_difference(n, refined, embedded; match_embedded_domain = false)
    data = fixture(n, refined, embedded, (1.0, 1.0); match_embedded_domain)
    (; mesh, geometry, operator, volumes) = data
    bottom = embedded || match_embedded_domain ? 1.0 : 0.0
    height = 8.0 - bottom
    a = pi / height
    # psi = sin(K*x) * sin(a*(y-bottom))^2 is continuously
    # divergence-free and vanishes with its normal derivative at both walls.
    velocity = FaceVelocity(mesh)
    for (i, face) in pairs(mesh.faces)
        volumes[i] > 0 || continue
        x, y = face.center
        s, c = sincos(a * (y - bottom))
        velocity.interior[i] = face.axis == 1 ?
            sin(K * x) * (2a * s * c) : -K * cos(K * x) * s^2
    end
    velocity.interior[operator.dirichlet_faces] .= 0
    scalar = embedded ?
        StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary = NoSlip()) :
        StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary = NoSlip())
    scalar_rate, variable_rate = FaceVelocity(mesh), FaceVelocity(mesh)
    diffusion!(scalar_rate, velocity, scalar)
    diffusion!(variable_rate, velocity, operator)
    difference = variable_rate.interior - scalar_rate.interior
    return ntuple(axis -> component_norm(mesh, volumes, difference, axis), 2)
end

rates(errors) = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]

@testset "two-phase embedded-wall manufactured Helmholtz (#106)" begin
    # The piecewise streamfunction gives div(u)=0, no slip at the two walls,
    # continuous velocity, and continuous nonzero tangential traction at the
    # material interface. Thus f=u-mu*Delta(u) has no omitted delta force.
    # At z=-1/2, Y=27/32, Y'=9/8, Y''=-21/4, and Y'''=6
    # for a_low=2. These values independently pin the signs and factors in
    # both sampled velocities and their analytic Laplacians.
    x, middle, halfheight = 1.0, 4.5, 3.5
    u1, lap1 = layer_value_laplacian(x, middle - halfheight / 2, 1, middle, halfheight, 2.0)
    u2, lap2 = layer_value_laplacian(x, middle - halfheight / 2, 2, middle, halfheight, 2.0)
    # Only a few elementary operations are involved; allow round-off.
    @test isapprox(u1, sin(K * x) * 9 / (8halfheight); rtol = 128eps(), atol = 128eps())
    @test isapprox(u2, -K * cos(K * x) * 27 / 32; rtol = 128eps(), atol = 128eps())
    @test isapprox(
        lap1, sin(K * x) * (6 / halfheight^3 - 9K^2 / (8halfheight));
        rtol = 128eps(), atol = 128eps(),
    )
    @test isapprox(
        lap2, -K * cos(K * x) * (-21 / (4halfheight^2) - 27K^2 / 32);
        rtol = 128eps(), atol = 128eps(),
    )
    interface_value, interface_force = exact_value_forcing(
        x, middle, 2, middle, halfheight, (2.0, 1.0),
    )
    @test isapprox(interface_value, -K * cos(K * x); rtol = 128eps(), atol = 128eps())
    @test isapprox(
        interface_force, -K * cos(K * x) * (1 + K^2);
        rtol = 128eps(), atol = 128eps(),
    )

    for embedded in (false, true), viscosities in ((1.0, 1.0), (2.0, 1.0))
        middle, halfheight = embedded ? (4.5, 3.5) : (4.0, 4.0)
        low, high = viscosities
        a_high = 2 + halfheight^2 * K^2 * (low / high - 1) / 2
        low_profile = profile(0.0, 2.0)
        high_profile = profile(0.0, a_high)
        @test low_profile[1:2] == high_profile[1:2]
        traction_low = low * (low_profile[3] / halfheight^2 + K^2 * low_profile[1])
        traction_high = high * (high_profile[3] / halfheight^2 + K^2 * high_profile[1])
        # A few operations on unit-scale analytic coefficients: round-off only.
        @test isapprox(traction_low, traction_high; rtol = 128eps(), atol = 128eps())
        @test traction_low > 0.5
        for axis in 1:2
            bottom = embedded ? 1.0 : 0.0
            for y in (bottom, 8.0)
                value, _ = exact_value_forcing(1.0, y, axis, middle, halfheight, viscosities)
                @test abs(value) <= 128eps()
            end
        end
    end

    for viscosities in ((1.0, 1.0), (2.0, 1.0)), refined in (false, true)
        # The degree-six streamfunction is pre-asymptotic on the coarsest
        # uniform grid, even without a solid; resolve it before measuring EOC.
        resolutions = viscosities == (1.0, 1.0) && !refined ?
            (32, 64, 128) : RESOLUTIONS
        wall = [helmholtz_errors(n, refined, true, viscosities) for n in resolutions]
        open = [helmholtz_errors(n, refined, false, viscosities) for n in resolutions]
        for axis in 1:2
            wall_errors = [result.errors[axis] for result in wall]
            open_errors = [result.errors[axis] for result in open]
            wall_rates = rates(wall_errors)
            open_rates = rates(open_errors)
            println(
                "embedded_variable_wall_mms viscosity=$viscosities refined=$refined ",
                "axis=$axis n=$resolutions wall_errors=$wall_errors ",
                "open_errors=$open_errors wall_eoc=$wall_rates open_eoc=$open_rates",
            )
            @test all(isfinite, wall_errors) && all(>(0), wall_errors)
            @test all(isfinite, open_errors) && all(>(0), open_errors)
            @test all(diff(wall_errors) .< 0)
            @test all(diff(open_errors) .< 0)
            if viscosities == (1.0, 1.0) && !refined
                @test all(wall_rates .>= 1.9)
            else
                # Only three levels and different wall heights are compared:
                # permit 10% variation in each error-reduction factor, i.e.
                # log2(1.1) in EOC, not an O(1) loss of convergence order.
                @test all(wall_rates .+ log2(1.1) .>= open_rates)
            end
        end
    end
end

@testset "uniform-viscosity stress versus scalar diffusion (#106)" begin
    for refined in (false, true)
        wall = [operator_difference(n, refined, true) for n in RESOLUTIONS]
        # Compare the embedded channel to the same open domain and spacing,
        # rather than to a taller channel with a different manufactured field.
        open = refined ? [
                operator_difference(n, true, false; match_embedded_domain = true)
                for n in RESOLUTIONS
            ] : nothing
        for axis in 1:2
            differences = [result[axis] for result in wall]
            wall_rates = rates(differences)
            println(
                "embedded_variable_wall_operator_difference refined=$refined axis=$axis ",
                "n=$RESOLUTIONS l2=$differences eoc=$wall_rates",
            )
            @test all(isfinite, differences) && all(>(0), differences)
            if refined
                open_errors = [result[axis] for result in open]
                open_rates = rates(open_errors)
                println(
                    "unembedded_variable_wall_operator_difference axis=$axis ",
                    "n=$RESOLUTIONS l2=$open_errors eoc=$open_rates"
                )
                # Compare actual defects on identical physical domains, not
                # just the pre-asymptotic EOC on two different-height domains.
                @test all(differences .<= 1.1 .* open_errors)
                @test all(diff(differences) .< 0)
                @test wall_rates[end] + log2(1.1) >= open_rates[end]
            else
                # Both operators are second order on the uniform channel, so
                # their difference converges at second order (from below).
                @test all(wall_rates .>= 1.9)
            end
        end
    end
end

@testset "two-phase aligned-wall constraints and reactions (#106)" begin
    for refined in (false, true)
        data = fixture(16, refined, true, (2.0, 1.0))
        (; mesh, geometry, interface, operator) = data
        constrained = operator.dirichlet_faces
        @test issorted(constrained) && allunique(constrained)
        cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
        cached = TwoPhaseAMR._staggered_diffusion_operator(
            interface; viscosity = (2.0, 1.0), boundary = NoSlip(),
            construction_cache = cache,
        )
        reused = TwoPhaseAMR._staggered_diffusion_operator(
            interface; viscosity = (2.0, 1.0), boundary = NoSlip(),
            construction_cache = cache, reuse = cached,
        )
        @test reused.dirichlet_faces == constrained
        @test reused.wall_reaction_strain == operator.wall_reaction_strain
        @test reused.strain == operator.strain
        @test reused.stiffness == operator.stiffness
        wall_faces = [
            i for (i, face) in pairs(mesh.faces)
                if face.axis == 2 && face.center[2] == 1.0
        ]
        @test !isempty(wall_faces)
        @test all(i -> i in constrained, wall_faces)
        # Covered ghost faces touching the solid-side cells can enter a
        # near-wall shear stencil and must be eliminated as well.
        wall_solid_cells = Set(
            TwoPhaseAMR._get(geometry.open_fraction, mesh.faces[i].minus) == 0 ?
                mesh.faces[i].minus : mesh.faces[i].plus for i in wall_faces
        )
        covered_ghosts = [
            i for (i, face) in pairs(mesh.faces)
                if iszero(geometry.face_dual_volumes[i]) &&
                (face.minus in wall_solid_cells || face.plus in wall_solid_cells)
        ]
        @test !isempty(covered_ghosts)
        @test all(i -> i in constrained, covered_ghosts)
        free = setdiff(collect(eachindex(mesh.faces)), constrained)
        @test !isempty(free)
        # Unit rows are algebraic extensions, not physical diffusion terms.
        @test all(i -> operator.stiffness[i, i] == 1, constrained)
        @test operator.stiffness[constrained, constrained] ==
            spdiagm(0 => ones(length(constrained)))
        @test all(iszero, nonzeros(operator.stiffness[free, constrained]))
        @test all(iszero, nonzeros(operator.stiffness[constrained, free]))
        # Symmetric assembly differs at most by accumulated round-off.
        scale = max(1.0, maximum(abs, nonzeros(operator.stiffness)))
        @test isapprox(
            operator.stiffness, transpose(operator.stiffness);
            rtol = 0, atol = 1024eps() * scale,
        )

        clean = FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? sin(K * x) + 0.1y : cos(K * x) - 0.2y,
        )
        clean.interior[constrained] .= 0
        dirty = copy(clean)
        dirty.interior[constrained] .= 13
        clean_rate = FaceVelocity(mesh)
        dirty_rate = FaceVelocity(mesh)
        diffusion!(clean_rate, clean, operator)
        diffusion!(dirty_rate, dirty, operator)
        @test all(iszero, dirty_rate.interior[constrained])
        @test isapprox(
            dirty_rate.interior, clean_rate.interior;
            rtol = 1024eps(), atol = 1024eps() * max(1.0, norm(clean_rate.interior)),
        )

        dt = 0.1
        density = ones(length(mesh.faces))
        sample = [sin(0.31i) + cos(0.13i) for i in eachindex(mesh.faces)]
        action = TwoPhaseAMR.FactorizedVariableDiffusionOperator(
            operator, density .* operator.volumes, dt,
        )
        factorized = similar(sample)
        mul!(factorized, action, sample)
        matrix = spdiagm(0 => density .* operator.volumes) + (dt / 2) * operator.stiffness
        assembled = matrix * sample
        @test isapprox(
            factorized, assembled;
            rtol = 1024eps(), atol = 1024eps() * max(1.0, norm(assembled)),
        )

        direct = copy(dirty)
        direct_clean = copy(clean)
        implicit_midpoint_diffusion!(direct, dirty, operator, dt; density)
        implicit_midpoint_diffusion!(direct_clean, clean, operator, dt; density)
        @test all(iszero, direct.interior[constrained])
        @test isapprox(
            direct.interior, direct_clean.interior;
            rtol = 1024eps(), atol = 1024eps() * max(1.0, norm(direct.interior)),
        )
        # Source and old velocity are independent arguments; neither may
        # reintroduce a prescribed value into the implicit right-hand side.
        for (source, old) in ((dirty, clean), (clean, dirty))
            mixed = copy(dirty)
            implicit_midpoint_diffusion!(
                mixed, source, operator, dt; old_velocity = old, density,
            )
            @test all(iszero, mixed.interior[constrained])
            @test isapprox(
                mixed.interior, direct_clean.interior;
                rtol = 1024eps(), atol = 1024eps() * max(1.0, norm(direct.interior)),
            )
        end

        # Use the default tree multigrid, including a deliberately dirty warm
        # start and source. Its Krylov tolerance is much smaller than the
        # solution-comparison tolerance (which also allows conditioning).
        solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(operator, density, dt)
        @test solver.preconditioner isa TwoPhaseAMR.TreeVariableDiffusionPreconditioner
        @test length(solver.preconditioner.levels) > 1
        multigrid = copy(dirty)
        implicit_midpoint_diffusion!(
            multigrid, dirty, operator, dt, solver;
            density, tolerance = 1.0e-11, max_cycles = 1000,
        )
        @test solver.stats.converged
        @test all(iszero, solver.right_hand_side[constrained])
        @test all(iszero, multigrid.interior[constrained])
        @test isapprox(multigrid.interior, direct.interior; rtol = 1.0e-8, atol = 1.0e-9)
        multigrid_clean = copy(dirty)
        implicit_midpoint_diffusion!(
            multigrid_clean, clean, operator, dt, solver;
            density, tolerance = 1.0e-11, max_cycles = 1000,
        )
        @test solver.stats.converged
        @test all(iszero, multigrid_clean.interior[constrained])
        @test isapprox(multigrid_clean.interior, direct_clean.interior; rtol = 1.0e-8, atol = 1.0e-9)
        for (source, old) in ((dirty, clean), (clean, dirty))
            mixed = copy(dirty)
            implicit_midpoint_diffusion!(
                mixed, source, operator, dt, solver;
                old_velocity = old, density, tolerance = 1.0e-11, max_cycles = 1000,
            )
            @test solver.stats.converged
            @test all(iszero, solver.right_hand_side[constrained])
            @test all(iszero, mixed.interior[constrained])
            @test isapprox(mixed.interior, direct_clean.interior; rtol = 1.0e-8, atol = 1.0e-9)
        end

        @test size(operator.wall_reaction_strain) ==
            (size(operator.strain, 1), length(constrained))
        # On a uniform grid, v=y-1 next to the bottom wall has normal stress
        # 2mu_low, hence its integrated upward wall reaction is 2mu_low*L.
        ramp = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 2 ?
                max(0.0, min(y - 1, 3 - y)) : 0.0,
        )
        ramp.interior[constrained] .= 0
        reaction(op) = -transpose(op.wall_reaction_strain) *
            (op.stress_weights .* (op.strain * ramp.interior))
        original_reaction = reaction(operator)
        wall_positions = findall(index -> index in wall_faces, constrained)
        @test length(wall_positions) == length(wall_faces)
        if !refined
            # The exact normal traction uses only linear data, so this is a
            # round-off tolerance proportional to the integrated force.
            expected = 2 * 2.0 * L
            @test isapprox(
                sum(original_reaction[wall_positions]), expected;
                rtol = 4096eps(), atol = 4096eps() * expected,
            )
        end
        @test norm(original_reaction) > 1
        old_strain = copy(operator.strain)
        old_reaction_strain = copy(operator.wall_reaction_strain)
        TwoPhaseAMR._update_dynamic_viscosity!(operator, interface, (6.0, 3.0))
        refreshed_reaction = reaction(operator)
        @test operator.strain == old_strain
        @test operator.wall_reaction_strain == old_reaction_strain
        # Every physical stress weight triples; unit constraint weights do not
        # contribute because the reaction metadata has zero algebraic rows.
        @test isapprox(
            refreshed_reaction, 3 .* original_reaction;
            rtol = 4096eps(), atol = 4096eps() * max(1.0, norm(refreshed_reaction)),
        )
        @test all(i -> operator.stiffness[i, i] == 1, constrained)
        rebuilt = StaggeredDiffusionOperator(
            interface; viscosity = (6.0, 3.0), boundary = NoSlip(),
        )
        @test rebuilt.dirichlet_faces == constrained
        @test rebuilt.wall_reaction_strain == operator.wall_reaction_strain
        @test isapprox(
            reaction(rebuilt), refreshed_reaction;
            rtol = 4096eps(), atol = 4096eps() * max(1.0, norm(refreshed_reaction)),
        )

        # Change the two viscosities by different factors as well. Check a
        # normal-stress weight in each layer and the full rebuilt operator,
        # not only the reaction near the lower wall.
        TwoPhaseAMR._update_dynamic_viscosity!(operator, interface, (1.5, 4.0))
        refs = TwoPhaseAMR._cell_references(mesh)
        for (ymin, ymax, viscosity) in ((1.0, 4.5, 1.5), (4.5, 8.0, 4.0))
            cell = first(
                i for (i, ref) in pairs(refs)
                    if ymin < cellcenter(mesh, ref.block, ref.i, ref.j)[2] < ymax
            )
            row = 2cell - 1
            @test isapprox(
                operator.stress_weights[row], viscosity * operator.stress_measures[row];
                rtol = 512eps(), atol = 512eps() * operator.stress_measures[row],
            )
        end
        changed = StaggeredDiffusionOperator(
            interface; viscosity = (1.5, 4.0), boundary = NoSlip(),
        )
        @test changed.dirichlet_faces == constrained
        @test changed.wall_reaction_strain == operator.wall_reaction_strain
        @test isapprox(
            operator.stress_weights, changed.stress_weights;
            rtol = 4096eps(), atol = 4096eps() * max(1.0, norm(changed.stress_weights)),
        )
        @test isapprox(
            operator.stiffness, changed.stiffness;
            rtol = 4096eps(), atol = 4096eps() * max(1.0, norm(changed.stiffness)),
        )
        @test isapprox(
            reaction(operator), reaction(changed);
            rtol = 4096eps(), atol = 4096eps() * max(1.0, norm(reaction(changed))),
        )
    end
end

@testset "coupled wall shear and intrinsic phase constraints (#106)" begin
    for refined in (false, true)
        data = fixture(16, refined, true, (2.0, 1.0))
        (; mesh, geometry, interface, operator) = data
        ramp = FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? max(0.0, min(y - 1, 3 - y)) : 0.0,
        )
        ramp.interior[operator.dirichlet_faces] .= 0
        force = operator.stiffness * ramp.interior
        tangential = [i for (i, face) in pairs(mesh.faces) if face.axis == 1]
        # Unit wall shear, viscosity 2, and periodic wall length 8.
        @test sum(force[tangential]) ≈ 16.0 atol = 2.0e-11

        free_slip = StaggeredDiffusionOperator(
            interface; viscosity = (2.0, 1.0), boundary = FreeSlip(),
        )
        constant = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 && y > 1 ? 1.0 : 0.0,
        )
        constant.interior[free_slip.dirichlet_faces] .= 0
        rate = FaceVelocity(mesh)
        diffusion!(rate, constant, free_slip)
        physical = findall(>(0), geometry.face_dual_volumes)
        @test maximum(abs, rate.interior[physical]) <= 2.0e-11

        source = (
            FaceVelocity(mesh, (x, y, time, axis) -> 2.0),
            FaceVelocity(mesh, (x, y, time, axis) -> 3.0),
        )
        destination = (FaceVelocity(mesh), FaceVelocity(mesh))
        first_wall, last_wall = first(operator.dirichlet_faces), last(operator.dirichlet_faces)
        source[1].interior[first_wall] = NaN
        source[2].interior[last_wall] = NaN
        mixture = FaceVelocity(mesh, (x, y, time, axis) -> 0.25)
        diffused = FaceVelocity(mesh, (x, y, time, axis) -> 0.75)
        TwoPhaseAMR._apply_two_velocity_diffusion_update!(
            destination, source, diffused, mixture;
            dirichlet_faces = operator.dirichlet_faces,
        )
        for phase in 1:2, index in operator.dirichlet_faces
            if isnan(source[phase].interior[index])
                @test isnan(destination[phase].interior[index])
            else
                @test iszero(destination[phase].interior[index])
            end
        end
        interior_face = first(
            i for i in eachindex(mesh.faces) if !(i in operator.dirichlet_faces)
        )
        @test destination[1].interior[interior_face] == 2.5
        @test destination[2].interior[interior_face] == 3.5
    end
end

@testset "Float32 variable-stress aligned wall (#106)" begin
    T = Float32
    mesh = BlockTreeMesh(;
        base_mesh = (8, 8), block_size = (4, 4),
        origin = (zero(T), zero(T)), extent = (T(8), T(8)),
        periodic = (true, false),
    )
    geometry = GeometryField(mesh, (x, y) -> y - T(4); lipschitz = one(T))
    interface = InterfaceField(mesh; geometry)
    initialize!(interface, (zero(T), one(T)), T(6))
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (one(T), one(T)), boundary = NoSlip(),
    )
    wall = [
        i for (i, face) in pairs(mesh.faces)
            if face.axis == 2 && face.center[2] == T(4)
    ]
    @test !isempty(wall)
    @test all(i -> i in operator.dirichlet_faces, wall)
    @test eltype(operator.stiffness) == T
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? max(zero(T), y - T(4)) : zero(T),
    )
    tendency = FaceVelocity(mesh)
    diffusion!(tendency, velocity, operator)
    face_index = findfirst(
        face -> face.axis == 1 && face.center == (T(2), T(4.5)), mesh.faces,
    )
    @test !isnothing(face_index)
    @test abs(tendency.interior[face_index]) <= T(64) * eps(T)
    @test all(iszero, tendency.interior[wall])
end

@testset "variable wall force and normal constraints in all orientations (#106)" begin
    for T in (Float32, Float64), normal in 1:2, side in (-1, 1),
            topology in (:uniform, :tangent_transition, :normal_transition)
        tangent = 3 - normal
        refinement_axis = topology == :tangent_transition ? tangent : normal
        mesh = BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (4, 4),
            origin = (zero(T), zero(T)), extent = (T(8), T(8)),
            periodic = normal == 1 ? (false, true) : (true, false), max_level = 1,
            refine = (key, origin, extent) ->
            topology != :uniform && key.level == 0 && origin[refinement_axis] >= T(4),
        )
        geometry = GeometryField(
            mesh, (x, y) -> T(side) * ((normal == 1 ? x : y) - T(4));
            lipschitz = one(T),
        )
        viscosity = CellField(mesh; init = T(0.3))
        wall_faces = [
            i for (i, face) in pairs(mesh.faces)
                if face.axis == normal && face.center[normal] == T(4)
        ]
        @test !isempty(wall_faces)
        for boundary in (NoSlip(), FreeSlip())
            operator = TwoPhaseAMR._staggered_diffusion_operator(
                mesh; viscosity, geometry, boundary,
            )
            @test all(i -> i in operator.dirichlet_faces, wall_faces)
            @test operator.stiffness ≈ transpose(operator.stiffness)
            wall_positions = findall(i -> i in wall_faces, operator.dirichlet_faces)
            @test length(wall_positions) == length(wall_faces)
            for axis in 1:2
                velocity = FaceVelocity(
                    mesh, (x, y, time, component) -> begin
                        d = T(side) * ((normal == 1 ? x : y) - T(4))
                        component == axis ? max(zero(T), min(d, T(2) - d)) : zero(T)
                    end,
                )
                velocity.interior[operator.dirichlet_faces] .= zero(T)
                force = operator.stiffness * velocity.interior
                faces = [i for (i, face) in pairs(mesh.faces) if face.axis == axis]
                expected = axis == normal ? T(2.0) * T(0.3) * T(8) :
                    boundary isa NoSlip ? T(0.3) * T(8) : zero(T)
                tolerance = T(4096) * eps(T) * T(8)
                # A tangential AMR shear fit can include covered faces, giving
                # a pre-existing truncation error even for a linear profile.
                # The force--reaction balance below is still checked exactly.
                physical_rtol = axis == normal && topology == :tangent_transition ?
                    max(tolerance, T(1.0e-3)) : tolerance
                @test isapprox(
                    sum(force[faces]), expected;
                    atol = tolerance, rtol = physical_rtol,
                )
                if axis == normal
                    reaction = -transpose(operator.wall_reaction_strain) *
                        (operator.stress_weights .* (operator.strain * velocity.interior))
                    normal_positions = findall(
                        i -> mesh.faces[operator.dirichlet_faces[i]].axis == normal,
                        eachindex(operator.dirichlet_faces),
                    )
                    @test isapprox(
                        sum(reaction[wall_positions]), expected;
                        atol = tolerance, rtol = physical_rtol,
                    )
                    # Elimination reactions include covered ghost faces when
                    # an AMR shear stencil reaches across the wall.
                    @test isapprox(
                        sum(force[faces]), sum(reaction[normal_positions]);
                        atol = tolerance, rtol = tolerance,
                    )
                end
            end
        end
    end
end

@testset "parallel PLIC walls constrain the solid-side normal face (#106)" begin
    for normal in 1:2, side in (-1, 1),
            topology in (:uniform, :tangent_transition, :normal_transition)
        tangent = 3 - normal
        refinement_axis = topology == :tangent_transition ? tangent : normal
        mesh = BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = normal == 1 ? (false, true) : (true, false), max_level = 1,
            refine = (key, origin, extent) ->
            topology != :uniform && key.level == 0 && origin[refinement_axis] >= 4,
        )
        wall = 4 + side / 4
        geometry = GeometryField(
            mesh, (x, y) -> side * ((normal == 1 ? x : y) - wall);
            lipschitz = 1,
        )
        viscosity = CellField(mesh; init = 1.0)
        solid_side = [
            i for (i, face) in pairs(mesh.faces)
                if face.axis == normal && face.center[normal] == 4 &&
                geometry.face_dual_volumes[i] > 0
        ]
        fluid_side = [
            i for (i, face) in pairs(mesh.faces)
                if face.axis == normal && side * (face.center[normal] - wall) > 0 &&
                geometry.aperture[i] > 0
        ]
        @test !isempty(solid_side)
        @test !isempty(fluid_side)
        @test all(i -> iszero(geometry.aperture[i]), solid_side)
        for boundary in (NoSlip(), FreeSlip())
            operator = TwoPhaseAMR._staggered_diffusion_operator(
                mesh; viscosity, geometry, boundary,
            )
            @test all(i -> i in operator.dirichlet_faces, solid_side)
            @test all(i -> !(i in operator.dirichlet_faces), fluid_side)
            velocity = FaceVelocity(mesh, (x, y, time, axis) -> 1.0)
            velocity.interior[solid_side] .= 7.0
            reference = copy(velocity)
            reference.interior[solid_side] .= 0.0
            result, clean = FaceVelocity(mesh), FaceVelocity(mesh)
            diffusion!(result, velocity, operator)
            diffusion!(clean, reference, operator)
            @test all(iszero, result.interior[solid_side])
            @test result.interior ≈ clean.interior
        end
    end
end

@testset "aligned variable-wall metadata survives AMR rebuilds (#106)" begin
    mesh = BlockTreeMesh(;
        block_size = (8, 8), extent = (8.0, 8.0),
        periodic = (false, false), max_level = 1,
    )
    geometry = GeometryField(mesh, (x, y) -> y - 4; lipschitz = 1)
    interface = InterfaceField(mesh, (x, y) -> x - 3.5; geometry, lipschitz = 1)
    cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    original = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = (2.0, 1.0), boundary = NoSlip(),
        construction_cache = cache,
    )
    @test !isempty(original.dirichlet_faces)
    adaptation = AMRWorkspace(mesh, interface, geometry)
    @test adapt!(adaptation, [:refine]; step = 1, force = true).changed
    reused = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = (2.0, 1.0), boundary = NoSlip(),
        construction_cache = cache, reuse = original,
    )
    fresh = StaggeredDiffusionOperator(interface; viscosity = (2.0, 1.0), boundary = NoSlip())
    @test !isempty(reused.dirichlet_faces)
    @test reused.dirichlet_faces == fresh.dirichlet_faces
    @test reused.wall_reaction_strain == fresh.wall_reaction_strain
    @test reused.strain == fresh.strain
    @test reused.stress_weights == fresh.stress_weights
    @test reused.stiffness == fresh.stiffness

    # The same persistent cache may subsequently construct an unembedded
    # operator and then return to the embedded geometry.
    viscosity = CellField(mesh; init = 1.0)
    unembedded = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, boundary = NoSlip(), construction_cache = cache,
        reuse = reused,
    )
    open_reference = StaggeredDiffusionOperator(mesh; viscosity, boundary = NoSlip())
    @test isempty(unembedded.dirichlet_faces)
    @test unembedded.strain == open_reference.strain
    @test unembedded.stiffness == open_reference.stiffness
    restored = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = (2.0, 1.0), boundary = NoSlip(),
        construction_cache = cache, reuse = unembedded,
    )
    @test restored.dirichlet_faces == fresh.dirichlet_faces
    @test restored.wall_reaction_strain == fresh.wall_reaction_strain
    @test restored.stiffness == fresh.stiffness
end

@testset "two-velocity diffusion substep prescribes aligned wall faces (#106)" begin
    mesh = BlockTreeMesh(;
        block_size = (8, 8), extent = (8.0, 8.0), periodic = (false, false),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 4; lipschitz = 1)
    interface = InterfaceField(mesh, (x, y) -> x - 3.5; geometry, lipschitz = 1)
    flow = FlowSimulation(
        mesh, TwoPhase(interface; density = (5.0, 1.0), dynamic_viscosity = (0.3, 0.1));
        velocity_model = TwoVelocityModel(), geometry, boundary = NoSlip(),
        velocity = (
            (x, y, time, axis) -> 1.0,
            (x, y, time, axis) -> 2.0,
        ),
        numerics = NavierStokesOptions(diffusion = LinearSolverOptions(preconditioner = nothing)),
    )
    model = flow.system.two_phase
    state = TwoPhaseAMR._two_velocity_flow_workspace!(flow.workspace.two_phase, model)
    TwoPhaseAMR._copy_velocity!(state.convected, flow.velocity)
    wall_faces = [
        i for (i, face) in pairs(mesh.faces)
            if face.axis == 2 && face.center[2] == 4
    ]
    @test !isempty(wall_faces)
    for phase in 1:2, index in wall_faces
        if isfinite(state.convected[phase].interior[index])
            state.convected[phase].interior[index] = 2.0 + phase
        end
    end
    TwoPhaseAMR._diffusion_substep!(
        flow.velocity, flow.system, flow.workspace, 0.01, 0.0, model;
        tolerance = 1.0e-10, max_cycles = 1000,
    )
    @test flow.workspace.two_phase.diffusion_solver.stats.converged
    for phase in 1:2, index in wall_faces
        if isfinite(state.convected[phase].interior[index])
            @test iszero(flow.velocity[phase].interior[index])
        else
            @test isnan(flow.velocity[phase].interior[index])
        end
    end
end

end # module EmbeddedVariableWallMMS
