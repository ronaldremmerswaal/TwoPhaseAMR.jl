using Test, TwoPhaseAMR

function droplet_energy(flow)
    density = mixture_density(
        flow.solver.two_phase.interface,
        flow.solver.two_phase.density
    )
    energy = 0.0
    for (index, face) in pairs(flow.solver.mesh.faces)
        hm = flow.solver.mesh.blocks[
            face.minus.block,
        ].spacing[face.axis]
        hp = flow.solver.mesh.blocks[
            face.plus.block,
        ].spacing[face.axis]
        dual_volume = face.area * (hm + hp) / 2
        energy += density[index] * dual_volume *
            flow.velocity.interior[index]^2 / 2
    end
    return energy
end

function _floating_droplet_mesh(kind, resolution)
    kind in (:uniform, :mixed, :amr) ||
        throw(ArgumentError("unknown mesh kind $kind"))
    if kind == :uniform
        return BlockTreeMesh(
            block_size = (resolution, resolution),
            periodic = (false, false)
        )
    end
    resolution % 4 == 0 ||
        throw(ArgumentError("mixed-grid resolution must be divisible by 4"))
    base = resolution ÷ 4
    return BlockTreeMesh(
        block_size = (base, base), max_level = 2,
        periodic = (false, false),
        refine = (key, origin, extent) -> kind == :mixed &&
            (
            key.level == 0 ||
                (key.level == 1 && key.i == key.j)
        )
    )
end

function _energy_peaks(times, energies)
    threshold = 0.05maximum(energies)
    return [
        index for index in 2:(length(energies) - 1)
            if energies[index] > energies[index - 1] &&
            energies[index] >= energies[index + 1] &&
            energies[index] > threshold
    ]
end

"""Thesis floating-droplet experiment.

The interface starts as an accurately integrated circle at rest. Discrete
curvature error supplies the infinitesimal perturbation; no phase-weighted
state or hand-built shape perturbation is introduced. On the square domain
the first symmetry-compatible mode is `k=4`, so kinetic energy has period
`T_sigma(4)/2`.
"""
function floating_droplet_case(kind, resolution; laplace = Inf)
    radius = 0.2
    center = (0.5, 0.5)
    mode = 4
    density = (1.0, 1.0e-3)
    sigma = 1.0
    viscosity = isinf(laplace) ? (0.0, 0.0) : begin
            liquid = sqrt(sigma * density[1] * radius / laplace)
            (liquid, liquid * sqrt(density[2] / density[1]))
        end
    levelset = (x, y) -> hypot(x - center[1], y - center[2]) - radius
    mesh = _floating_droplet_mesh(kind, resolution)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    if kind == :amr
        indicator = InterfaceRefinementIndicator(interface; level = 2)
        adaptation = AMRWorkspace(mesh, interface)
        for step in 1:2
            adapt!(adaptation, indicator; step, force = true)
            initialize!(interface, levelset; lipschitz = 1)
        end
    end
    initial_fraction = copy(interface.fraction)
    initial_volume = integrate(interface)
    flow = FlowSimulation(
        mesh, TwoPhase(interface; dynamic_viscosity = viscosity, density,
            surface_tension = SurfaceTension(sigma; method = CGFM()));
        boundary = FreeSlip(),
    )
    shape_period = 2pi * sqrt(
        sum(density) * radius^3 /
            (sigma * mode * (mode + 1) * (mode - 1))
    )
    energy_period = shape_period / 2
    final_time = 1.3shape_period
    dt = min(shape_period / 320, 0.8stable_timestep(flow))
    times = Float64[0]
    energies = Float64[0]
    while flow.time < final_time - 16eps(final_time)
        target = min(final_time, flow.time + dt)
        simulate!(
            flow, target; dt = target - flow.time,
            tolerance = 1.0e-9, max_cycles = 200
        )
        push!(times, flow.time)
        push!(energies, droplet_energy(flow))
    end
    peaks = _energy_peaks(times, energies)
    measured_period = length(peaks) < 2 ? NaN :
        sum(diff(times[peaks])) / (length(peaks) - 1)
    shape_change = sum(
        mesh.blocks[ref.block].volume * abs(
            TwoPhaseAMR._get(interface.fraction, ref) -
                TwoPhaseAMR._get(initial_fraction, ref)
        ) for ref in TwoPhaseAMR._cell_references(mesh)
    )
    return (;
        measured_period, reference_period = energy_period,
        maximum_energy = maximum(energies),
        peak_energies = energies[peaks], shape_change,
        volume_error = abs(integrate(interface) - initial_volume),
        volume_tolerance = interface.fraction_floor * prod(mesh.extent),
        cells = ncells(mesh),
    )
end

@testset "Floating-droplet integration" begin
    @testset "Floating droplet: equilibrium and Lamb frequency" begin
        for laplace in (Inf, 1.0e3)
            coarse = floating_droplet_case(
                :uniform, 16; laplace
            )
            fine = floating_droplet_case(
                :uniform, 32; laplace
            )
            coarse_frequency_error = abs(
                coarse.measured_period - coarse.reference_period
            )
            fine_frequency_error = abs(
                fine.measured_period - fine.reference_period
            )
            @test isfinite(coarse.measured_period)
            @test isfinite(fine.measured_period)
            @test fine_frequency_error < coarse_frequency_error / 4
            @test fine.maximum_energy < coarse.maximum_energy / 4
            # Wisp removal deliberately snaps sub-fraction-floor phase
            # volumes. Refinement can therefore expose more floor-scale
            # changes even when the geometric transport remains conservative.
            @test fine.volume_error <=
                coarse.volume_error + fine.volume_tolerance
            if isfinite(laplace)
                @test length(fine.peak_energies) >= 2
                @test last(fine.peak_energies) < first(fine.peak_energies)
            end
        end

        # Exercise the same physical problem on fixed mixed-level and
        # dynamically interface-refined trees. Uniform-grid convergence above
        # supplies the reference-order assertion; these cases catch transfer
        # and coarse/fine operator failures without comparing to a loose
        # problem-specific error constant.
        for kind in (:mixed, :amr), laplace in (Inf, 1.0e3)
            result = floating_droplet_case(kind, 32; laplace)
            @test isfinite(result.maximum_energy)
            @test isfinite(result.shape_change)
            @test result.volume_error <= result.volume_tolerance
        end
    end
end
