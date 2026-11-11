using Test, TwoPhaseAMR

"""Independent periodic 1-D form of one directional WY phase/scalar sweep."""
function reference_wy_sweep(
        phase, scalar, indicator, carrier_flux, phase_flux;
        volumes = ones(eltype(phase), length(phase)),
    )
    n = length(phase)
    all(
        length(array) == n for array in (
                scalar, indicator, carrier_flux, phase_flux, volumes,
            )
    ) || throw(DimensionMismatch("all reference arrays must have equal length"))
    amount = phase .* scalar
    next_phase = similar(phase)
    next_amount = similar(amount)
    for cell in 1:n
        left = cell == 1 ? n : cell - 1
        right = cell
        qleft, qright = carrier_flux[left], carrier_flux[right]
        fleft, fright = phase_flux[left], phase_flux[right]
        sleft = fleft >= 0 ? scalar[left] : scalar[cell]
        next = cell == n ? 1 : cell + 1
        sright = fright >= 0 ? scalar[cell] : scalar[next]
        dilation = indicator[cell] * (qright - qleft)
        next_phase[cell] = phase[cell] + (
            dilation - (fright - fleft)
        ) / volumes[cell]
        next_amount[cell] = amount[cell] + (
            indicator[cell] * scalar[cell] * (qright - qleft) -
                (fright * sright - fleft * sleft)
        ) / volumes[cell]
    end
    next_scalar = similar(scalar)
    for cell in 1:n
        next_scalar[cell] = iszero(next_phase[cell]) ? scalar[cell] :
            next_amount[cell] / next_phase[cell]
    end
    return next_phase, next_scalar, next_amount
end

"""Independent conservative auxiliary carrier used by projected Method A."""
function reference_conservative_sweep(
        phase, scalar, phase_flux;
        volumes = ones(eltype(phase), length(phase)),
    )
    n = length(phase)
    amount = phase .* scalar
    next_phase = similar(phase)
    next_amount = similar(amount)
    for cell in 1:n
        left = cell == 1 ? n : cell - 1
        right = cell
        fleft, fright = phase_flux[left], phase_flux[right]
        next = cell == n ? 1 : cell + 1
        sleft = fleft >= 0 ? scalar[left] : scalar[cell]
        sright = fright >= 0 ? scalar[cell] : scalar[next]
        next_phase[cell] = phase[cell] -
            (fright - fleft) / volumes[cell]
        next_amount[cell] = amount[cell] -
            (fright * sright - fleft * sleft) / volumes[cell]
    end
    next_scalar = similar(scalar)
    for cell in 1:n
        next_scalar[cell] = iszero(next_phase[cell]) ? scalar[cell] :
            next_amount[cell] / next_phase[cell]
    end
    return next_phase, next_scalar, next_amount
end

@testset "Two-velocity directional WY reference equations" begin
    liquid = [0.0, 0.2, 0.8, 1.0, 0.65, 0.1]
    gas = 1 .- liquid
    indicator = Float64.(liquid .> 0.5)
    gas_indicator = 1 .- indicator

    # These signed arrays stand for geometrically integrated partial donating
    # regions. Their partition is the only assumption used by the balance.
    carrier = [0.015, -0.008, 0.012, 0.004, -0.006, -0.003]
    liquid_flux = [0.0, -0.002, 0.009, 0.004, -0.004, 0.0]
    gas_flux = carrier .- liquid_flux

    new_liquid, liquid_one, liquid_amount = reference_wy_sweep(
        liquid, ones(6), indicator, carrier, liquid_flux,
    )
    new_gas, gas_one, gas_amount = reference_wy_sweep(
        gas, ones(6), gas_indicator, carrier, gas_flux,
    )
    @test new_liquid + new_gas ≈ ones(6) atol = 8eps()
    @test liquid_one ≈ ones(6) atol = 8eps()
    @test gas_one ≈ ones(6) atol = 8eps()
    @test liquid_amount == new_liquid
    @test gas_amount == new_gas
    # A non-solenoidal directional carrier can exchange the two phase volumes
    # through WY dilation, but their sum remains the open volume. With a
    # discretely solenoidal (constant, in 1-D) carrier, each phase is globally
    # conservative as well.
    solenoidal_carrier = fill(0.01, 6)
    solenoidal_liquid_flux = [0.0, 0.002, 0.009, 0.01, 0.006, 0.0]
    solenoidal_gas_flux = solenoidal_carrier .- solenoidal_liquid_flux
    conservative_liquid = reference_wy_sweep(
        liquid, ones(6), indicator, solenoidal_carrier,
        solenoidal_liquid_flux,
    )[1]
    conservative_gas = reference_wy_sweep(
        gas, ones(6), gas_indicator, solenoidal_carrier,
        solenoidal_gas_flux,
    )[1]
    @test sum(conservative_liquid) ≈ sum(liquid) atol = 8eps()
    @test sum(conservative_gas) ≈ sum(gas) atol = 8eps()

    liquid_scalar = 2 .+ 0.2 .* (1:6)
    gas_scalar = 5 .- 0.1 .* (1:6)
    _, transported_liquid, _ = reference_wy_sweep(
        liquid, liquid_scalar, indicator, carrier, liquid_flux,
    )
    _, transported_gas, _ = reference_wy_sweep(
        gas, gas_scalar, gas_indicator, carrier, gas_flux,
    )
    @test all(isfinite, transported_liquid)
    @test all(isfinite, transported_gas)

    # The projected method's gas carrier is conservative and deliberately
    # need not equal the complement of the physical liquid fraction.
    auxiliary_gas, auxiliary_one, auxiliary_amount =
        reference_conservative_sweep(gas, ones(6), gas_flux)
    @test sum(auxiliary_gas) ≈ sum(gas) atol = 8eps()
    @test auxiliary_one ≈ ones(6) atol = 8eps()
    @test auxiliary_amount == auxiliary_gas
    @test maximum(abs, auxiliary_gas - new_gas) > 0

    # Equal-velocity reduction: splitting a common carrier by phase and using
    # the same WY equation is exactly the ordinary one-velocity algebra.
    common_liquid = liquid_flux
    common_gas = carrier .- common_liquid
    reduced_liquid = reference_wy_sweep(
        liquid, ones(6), indicator, carrier, common_liquid,
    )[1]
    reduced_gas = reference_wy_sweep(
        gas, ones(6), gas_indicator, carrier, common_gas,
    )[1]
    @test reduced_liquid == new_liquid
    @test reduced_gas == new_gas
end

# Independent polygon clipping for a normalized donor rectangle. This does
# not use the production area or swept-fraction routines.
function frozen_reference_open_area(field, donor, axis, nlo, nhi, tlo, thi)
    nhi > nlo && thi > tlo || return 0.0
    geometry = field.geometry
    isnothing(geometry) && return (nhi - nlo) * (thi - tlo)
    solid = geometry.solid
    fraction = TwoPhaseAMR._get(solid.fraction, donor)
    fraction <= 0 && return (nhi - nlo) * (thi - tlo)
    fraction >= 1 && return 0.0
    xlo, xhi, ylo, yhi = axis == 1 ?
        (nlo, nhi, tlo, thi) : (tlo, thi, nlo, nhi)
    polygon = [(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)]
    nx = TwoPhaseAMR._get(solid.normal_x, donor)
    ny = TwoPhaseAMR._get(solid.normal_y, donor)
    intercept = TwoPhaseAMR._get(solid.intercept, donor)
    signed(p) = nx * p[1] + ny * p[2] - intercept
    clipped = Tuple{Float64, Float64}[]
    previous = polygon[end]
    previous_value = signed(previous)
    for current in polygon
        current_value = signed(current)
        if (previous_value >= 0) != (current_value >= 0)
            ratio = previous_value / (previous_value - current_value)
            push!(
                clipped, (
                    previous[1] + ratio * (current[1] - previous[1]),
                    previous[2] + ratio * (current[2] - previous[2]),
                )
            )
        end
        current_value >= 0 && push!(clipped, current)
        previous, previous_value = current, current_value
    end
    length(clipped) >= 3 || return 0.0
    twice_area = 0.0
    for index in eachindex(clipped)
        a = clipped[index]
        b = clipped[index == length(clipped) ? 1 : index + 1]
        twice_area += a[1] * b[2] - a[2] * b[1]
    end
    return abs(twice_area) / 2
end

# Reconstruct the actual phase-subface donating rectangles by independently
# inverting their open areas. Check representation and all same-donor
# intersections, including different carriers on the same or opposite faces.
function frozen_reference_regions(field, velocity, work, dt)
    mesh = field.mesh
    regions = NamedTuple[]
    for (index, face) in pairs(mesh.faces), phase in 1:2
        aperture = phase == 1 ? work.liquid_aperture[index] :
            work.gas_aperture[index]
        aperture > 0 || continue
        u = velocity[phase].interior[index]
        iszero(u) && continue
        donor = u > 0 ? face.minus : face.plus
        tlo, thi = TwoPhaseAMR._frozen_subface_interval(
            field, work, face, index, donor, phase == 1, field.geometry,
        )
        target = abs(dt * u * face.area * aperture) /
            mesh.blocks[donor.block].volume
        bounds(depth) = u > 0 ? (0.5 - depth, 0.5) : (-0.5, -0.5 + depth)
        function area(depth)
            nlo, nhi = bounds(depth)
            return frozen_reference_open_area(field, donor, face.axis, nlo, nhi, tlo, thi)
        end
        # 512 eps in normalized full-cell units covers clipping round-off.
        @test target <= area(1.0) + 512eps()
        lower, upper = 0.0, 1.0
        for _ in 1:60
            middle = (lower + upper) / 2
            if area(middle) < target
                lower = middle
            else
                upper = middle
            end
        end
        depth = (lower + upper) / 2
        @test isapprox(area(depth), target; rtol = 0, atol = 512eps())
        nlo, nhi = bounds(depth)
        push!(regions, (; donor, index, axis = face.axis, nlo, nhi, tlo, thi, depth, u, phase))
    end
    for i in eachindex(regions), j in (i + 1):length(regions)
        a, b = regions[i], regions[j]
        a.donor == b.donor && a.axis == b.axis || continue
        nlo, nhi = max(a.nlo, b.nlo), min(a.nhi, b.nhi)
        tlo, thi = max(a.tlo, b.tlo), min(a.thi, b.thi)
        overlap = frozen_reference_open_area(field, a.donor, a.axis, nlo, nhi, tlo, thi)
        @test overlap <= 512eps()
    end
    return regions
end

function frozen_reference_bounded(field)
    for block in eachindex(field.mesh.blocks)
        conditional = interior(field.conditional_fraction, block)
        liquid = interior(field.fraction, block)
        open = isnothing(field.geometry) ? ones(size(liquid)) :
            interior(field.geometry.open_fraction, block)
        @test all(c -> 0 <= c <= 1, conditional)
        @test all(index -> -128eps() <= liquid[index] <= open[index] + 128eps(), eachindex(liquid))
    end
    return
end

function frozen_counterexample_fixture()
    mesh = BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (2, 2), extent = (4.0, 4.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key == BlockKey(0, 1, 1),
    )
    refs = CellRef[]
    for block in eachindex(mesh.blocks), j in 1:mesh.block_size[2], i in 1:mesh.block_size[1]
        cellcenter(mesh, block, i, j) == (1.5, 2.5) || continue
        push!(refs, CellRef(block, i, j))
    end
    donor = only(refs)
    geometry = GeometryField(mesh, (x, y) -> y + 100.0; lipschitz = 1)
    angle, offset = 6.047103058103431, 0.3519469972142466
    nx, ny = cos(angle), sin(angle)
    solid_fraction = TwoPhaseAMR._plane_fraction(nx, ny, offset, -0.5, 0.5, -0.5, 0.5)
    TwoPhaseAMR._set!(geometry.solid.fraction, solid_fraction, donor)
    TwoPhaseAMR._set!(geometry.solid.normal_x, nx, donor)
    TwoPhaseAMR._set!(geometry.solid.normal_y, ny, donor)
    TwoPhaseAMR._set!(geometry.solid.intercept, offset, donor)
    TwoPhaseAMR._set!(geometry.open_fraction, 1 - solid_fraction, donor)
    for (index, face) in pairs(mesh.faces)
        geometry.aperture[index] = TwoPhaseAMR._interior_open_aperture(geometry.solid, face)
    end
    liquid_angle, liquid_offset = 1.263045452558202, 0.16999975582492988
    field = InterfaceField(
        mesh,
        (x, y) -> cos(liquid_angle) * (x - 1.5) +
            sin(liquid_angle) * (y - 2.5) - liquid_offset;
        geometry, lipschitz = 1,
    )
    reconstruct!(field)
    right = [index for (index, face) in pairs(mesh.faces) if face.axis == 1 && face.minus == donor]
    left = [index for (index, face) in pairs(mesh.faces) if face.axis == 1 && face.plus == donor]
    @test length(right) == 2 && length(left) == 1
    liquid, gas = FaceVelocity(mesh), FaceVelocity(mesh)
    liquid.interior[right[1]] = -0.10191943684044136
    liquid.interior[left[1]] = -0.19799910106953192
    gas.interior[right[1]] = -0.10848167605356168
    gas.interior[right[2]] = 0.179524103894174
    return field, (liquid, gas), donor
end

@testset "Frozen subface cut-donor CFL and boundedness (#56)" begin
    # This cut coarse donor and hanging face produced C=1.207 at the former
    # stable step 0.40359. Its left half is closed, but its full strip has
    # open volume: a valid bound must not collapse to zero.
    for sweep_order in ((1, 2), (2, 1))
        field, velocity, donor = frozen_counterexample_fixture()
        work = WYAdvectionWorkspace(field, velocity)
        limit = stable_timestep(velocity, work)
        @test isfinite(limit) && 0 < limit < 0.40359
        dt = 0.99 * limit
        regions = frozen_reference_regions(field, velocity, work, dt)
        left_regions = [r for r in regions if r.donor == donor && r.axis == 1 && r.u < 0]
        @test !isempty(left_regions)
        @test any(left_regions) do r
            frozen_reference_open_area(field, donor, 1, -0.5, 0.0, r.tlo, r.thi) <= 512eps() &&
                r.depth > 0.5
        end

        # The PLIC is reconstructed between sweeps. The frozen subface must
        # retain its position even if the current normal changes sign.
        mixed = findfirst(
            i -> work.liquid_aperture[i] > 0 && work.gas_aperture[i] > 0,
            eachindex(field.mesh.faces)
        )
        @test !isnothing(mixed)
        face = field.mesh.faces[mixed]
        tangent = 3 - face.axis
        normals = tangent == 1 ? field.normal_x : field.normal_y
        old_minus = TwoPhaseAMR._get(normals, face.minus)
        old_plus = TwoPhaseAMR._get(normals, face.plus)
        interval = TwoPhaseAMR._frozen_subface_interval(
            field, work, face, mixed, face.minus, true, field.geometry,
        )
        opposite = work.liquid_subface_lower[mixed] ? -1.0 : 1.0
        TwoPhaseAMR._set!(normals, opposite, face.minus)
        TwoPhaseAMR._set!(normals, opposite, face.plus)
        @test TwoPhaseAMR._frozen_subface_interval(
            field, work, face, mixed, face.minus, true, field.geometry,
        ) == interval
        TwoPhaseAMR._set!(normals, old_minus, face.minus)
        TwoPhaseAMR._set!(normals, old_plus, face.plus)

        advect!(field, velocity, dt, work; sweep_order)
        frozen_reference_bounded(field)
    end
end

@testset "Frozen subface uses its own donor-open strip" begin
    mesh = BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (4, 4), extent = (4.0, 4.0),
        periodic = (false, false),
    )
    donor = CellRef(1, 2, 2)
    geometry = GeometryField(mesh, (x, y) -> y + 100.0; lipschitz = 1)
    # The donor's open part is its upper quarter; its neighbour is uncut.
    TwoPhaseAMR._set!(geometry.solid.fraction, 0.75, donor)
    TwoPhaseAMR._set!(geometry.solid.normal_x, 0.0, donor)
    TwoPhaseAMR._set!(geometry.solid.normal_y, 1.0, donor)
    TwoPhaseAMR._set!(geometry.solid.intercept, 0.25, donor)
    TwoPhaseAMR._set!(geometry.open_fraction, 0.25, donor)
    for (index, face) in pairs(mesh.faces)
        geometry.aperture[index] = TwoPhaseAMR._interior_open_aperture(geometry.solid, face)
    end
    index = findfirst(face -> face.axis == 1 && face.center == (2.0, 1.5), mesh.faces)
    @test !isnothing(index)
    face = mesh.faces[index]
    @test face.minus == donor
    # The pure-solid neighbour's normal has no effect on its geometry; set it
    # to select the same upper open interval as the donor's solid plane.
    TwoPhaseAMR._set!(geometry.solid.normal_y, 1.0, face.plus)
    field = InterfaceField(mesh; geometry)
    initialize!(field, (0.0, 1.0), 1.775)
    # Prescribe the exact horizontal liquid PLIC on both sides of this face.
    for ref in (face.minus, face.plus)
        TwoPhaseAMR._set!(field.normal_x, 0.0, ref)
        TwoPhaseAMR._set!(field.normal_y, 1.0, ref)
        TwoPhaseAMR._set!(field.intercept, 0.275, ref)
    end
    field.phase_apertures_valid[] = false
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    velocity[1].interior[index] = 1.0
    work = WYAdvectionWorkspace(field, velocity)
    limit = stable_timestep(velocity, work)
    @test work.liquid_aperture[index] ≈ 0.4 atol = 512eps()
    lo, hi = TwoPhaseAMR._frozen_subface_interval(
        field, work, face, index, donor, true, geometry,
    )
    strip = frozen_reference_open_area(field, donor, 1, -0.5, 0.5, lo, hi)
    @test strip ≈ 0.025 atol = 512eps()
    @test limit ≈ strip / work.liquid_aperture[index] atol = 512eps()
    # The old outgoing-volume bound was 0.3125, requesting five times this
    # subface's open strip despite satisfying the whole-cell CFL condition.
    @test 0.3125 * work.liquid_aperture[index] > 4strip
    dt = 0.99 * limit
    @test !isempty(frozen_reference_regions(field, velocity, work, dt))
    advect!(field, velocity, dt, work)
    frozen_reference_bounded(field)

    # A nonzero averaged aperture can also place a phase subface entirely in
    # solid, even over the *full* strip. No positive timestep represents that
    # prescribed carrier; the bound must not invent open volume.
    blocked = InterfaceField(mesh; geometry)
    initialize!(blocked, (0.0, 1.0), 1.575)
    for ref in (face.minus, face.plus)
        TwoPhaseAMR._set!(blocked.normal_x, 0.0, ref)
        TwoPhaseAMR._set!(blocked.normal_y, 1.0, ref)
        TwoPhaseAMR._set!(blocked.intercept, 0.075, ref)
    end
    blocked.phase_apertures_valid[] = false
    blocked_work = WYAdvectionWorkspace(
        blocked, velocity,
    )
    # No positive timestep represents that prescribed carrier; the bound must
    # not invent open volume, and it reports the limiting face and cell.
    error = try
        stable_timestep(velocity, blocked_work)
        nothing
    catch exception
        exception
    end
    @test error isa ArgumentError
    @test occursin("liquid subface", error.msg)
    @test occursin("(block 1, i 2, j 2)", error.msg)
    @test blocked_work.liquid_aperture[index] > 0
    lo, hi = TwoPhaseAMR._frozen_subface_interval(
        blocked, blocked_work, face, index, donor, true, geometry,
    )
    @test frozen_reference_open_area(blocked, donor, 1, -0.5, 0.5, lo, hi) == 0
end

@testset "Frozen same-face and opposing phase carriers" begin
    mesh = BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (4, 4), extent = (4.0, 4.0),
        periodic = (false, false),
    )
    field = InterfaceField(mesh)
    initialize!(field, (0.0, 1.0), 1.5)
    donor = CellRef(1, 2, 2)
    left = findfirst(face -> face.axis == 1 && face.center == (1.0, 1.5), mesh.faces)
    right = findfirst(face -> face.axis == 1 && face.center == (2.0, 1.5), mesh.faces)
    @test !isnothing(left) && !isnothing(right)
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    velocity[1].interior[left], velocity[1].interior[right] = -1.3, 1.0
    velocity[2].interior[left], velocity[2].interior[right] = -0.2, 2.0
    work = WYAdvectionWorkspace(field, velocity)
    limit = stable_timestep(velocity, work)
    @test isfinite(limit) && limit > 0
    dt = 0.99 * limit
    regions = frozen_reference_regions(field, velocity, work, dt)
    donor_carriers = Set((r.index, r.phase) for r in regions if r.donor == donor)
    @test donor_carriers == Set(((left, 1), (left, 2), (right, 1), (right, 2)))
    advect!(field, velocity, dt, work)
    frozen_reference_bounded(field)
end

@testset "Frozen subfaces on a sloped-solid touching drop" begin
    # Sessile-drop fixture of docs/techref/figures/vof.jl, including its
    # coarse/fine boundary and sloped wall.
    mesh = BlockTreeMesh(;
        base_mesh = (12, 12), block_size = (4, 4), extent = (12.0, 12.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key == BlockKey(0, 1, 1),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 2.2625 - 0.25x; lipschitz = hypot(1, 0.25))
    field = InterfaceField(
        mesh, (x, y) -> hypot(x - 6.4, y - 3.8625) - 2.62;
        geometry, lipschitz = 1,
    )
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    index = findfirst(face -> face.axis == 1 && face.center == (4.0, 3.5), mesh.faces)
    @test !isnothing(index)
    velocity[1].interior[index] = 0.3
    velocity[2].interior[index] = -0.2
    work = WYAdvectionWorkspace(field, velocity)
    limit = stable_timestep(velocity, work)
    @test isfinite(limit) && limit > 0
    dt = 0.99 * limit
    @test !isempty(frozen_reference_regions(field, velocity, work, dt))
    advect!(field, velocity, dt, work)
    frozen_reference_bounded(field)
end

@testset "Frozen projected carrier for the wavy-wall drop (#56)" begin
    # The adaptive two-velocity flow behind the ch. 12 observation.
    mesh = BlockTreeMesh(;
        base_mesh = (32, 16), block_size = (8, 8), extent = (2.0, 1.0),
        periodic = (false, false), max_level = 2,
        adaptation = MeshAdaptationOptions(coarsen_delay = 2),
    )
    wall(x, y) = y - 0.15 - 0.1sinpi(x)
    drop(x, y) = hypot(x - 0.7, y - 0.42) - 0.22
    geometry = GeometryField(mesh, wall; lipschitz = 1.5)
    interface = InterfaceField(mesh, drop; geometry, lipschitz = 1)
    indicators = (
        InterfaceRefinementIndicator(interface; level = 2),
        GeometryRefinementIndicator(geometry; level = 1),
    )
    adaptation = AMRWorkspace(mesh, interface, geometry)
    for pass in 1:3
        adapt!(adaptation, indicators; step = pass, force = true).changed || break
        initialize!(interface, drop; lipschitz = 1)
    end
    velocity(x, y, t, axis) = axis == 1 ?
        pi * sinpi(x / 2)^2 * sinpi(2y) :
        -pi / 2 * sinpi(x) * sinpi(y)^2
    physics = TwoPhase(
        interface; density = (1.0, 0.1), dynamic_viscosity = (1.0e-3, 1.0e-4),
        surface_tension = 0.01,
    )
    flow = FlowSimulation(
        mesh, physics; velocity_model = TwoVelocityModel(), velocity,
        geometry, boundary = NoSlip(), adaptivity = Adaptivity(criteria = indicators),
    )
    project!(flow; tolerance = 1.0e-12)
    for order in ((1, 2), (2, 1)), factor in (0.99, 1.0)
        field = copy(interface)
        work = WYAdvectionWorkspace(
            field, flow.velocity,
        )
        limit = stable_timestep(flow.velocity, work)
        @test isfinite(limit) && limit > 0
        advect!(field, flow.velocity, factor * limit, work; sweep_order = order)
        frozen_reference_bounded(field)
    end
end

@testset "Frozen empty apertures and uncut CFL" begin
    for T in (Float32, Float64)
        mesh = BlockTreeMesh(;
            base_mesh = (4, 4), block_size = (4, 4),
            origin = (zero(T), zero(T)), extent = (T(4), T(4)),
            periodic = (true, true),
        )
        field = InterfaceField(mesh; init = one(T))
        liquid = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? T(0.2) : zero(T))
        gas = FaceVelocity(mesh)
        fill!(gas.interior, T(NaN)) # The absent carrier must never be read.
        velocity = (liquid, gas)
        work = WYAdvectionWorkspace(field, velocity)
        limit = stable_timestep(velocity, work)
        @test limit isa T
        @test limit ≈ T(2.5) rtol = 32eps(T)
        @test stable_timestep(velocity, work; cfl = 0.25) ≈ limit / 2 rtol = 32eps(T)
        @test all(iszero, work.gas_aperture)
        dt = T(0.99) * limit
        @test !isempty(frozen_reference_regions(field, velocity, work, dt))
        advect!(field, velocity, dt, work)
        frozen_reference_bounded(field)
    end
end

@testset "Phase aperture freezing for both workspace constructors" begin
    mesh = BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (4, 4), extent = (4.0, 4.0),
        periodic = (false, false),
    )
    field = InterfaceField(mesh)
    initialize!(field, (0.0, 1.0), 1.5)
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    workspaces = (
        WYAdvectionWorkspace(field),
        WYAdvectionWorkspace(field, velocity),
    )
    for work in workspaces
        @test isempty(work.liquid_subface_lower)
        TwoPhaseAMR._freeze_phase_apertures!(field, work)
        @test work.liquid_aperture == field.phase_liquid_aperture
        @test work.gas_aperture == field.phase_gas_aperture
    end

    index = findfirst(face -> face.axis == 1 && face.center == (2.0, 1.5), mesh.faces)
    @test !isnothing(index)
    face = mesh.faces[index]

    # Without a solid, subface ordering continues to follow the current PLIC
    # and requires no cut-geometry storage.
    uncut = WYAdvectionWorkspace(field, velocity)
    TwoPhaseAMR._freeze_phase_apertures!(field, uncut)
    @test isempty(uncut.liquid_subface_lower)
    uncut.liquid_aperture[index] = 0.25
    uncut.gas_aperture[index] = 0.75
    for ref in (face.minus, face.plus)
        TwoPhaseAMR._set!(field.normal_y, 1.0, ref)
    end
    lower = TwoPhaseAMR._frozen_subface_interval(field, uncut, face, index, face.minus, true, nothing)
    for ref in (face.minus, face.plus)
        TwoPhaseAMR._set!(field.normal_y, -1.0, ref)
    end
    upper = TwoPhaseAMR._frozen_subface_interval(field, uncut, face, index, face.minus, true, nothing)
    @test lower[2] < upper[1]

    # Even a tiny partial aperture paired with a nominally full opposite
    # aperture needs the frozen ordering when a solid is present.
    geometry = GeometryField(mesh, (x, y) -> y - 0.25; lipschitz = 1)
    cut = InterfaceField(mesh; geometry)
    initialize!(cut, (0.0, 1.0), 1.5)
    frozen = WYAdvectionWorkspace(cut, velocity)
    TwoPhaseAMR._freeze_phase_apertures!(cut, frozen)
    for ref in (face.minus, face.plus)
        TwoPhaseAMR._set!(cut.normal_y, 1.0, ref)
    end
    cut.phase_liquid_aperture[index] = eps() / 4
    cut.phase_gas_aperture[index] = 1.0
    TwoPhaseAMR._freeze_phase_apertures!(cut, frozen)
    @test frozen.liquid_subface_lower[index]
end
