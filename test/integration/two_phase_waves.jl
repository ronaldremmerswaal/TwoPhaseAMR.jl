using Test, TwoPhaseAMR
function wave_amplitude(f)
    mesh = f.mesh;result = 0.0
    for ref in TwoPhaseAMR._cell_references(mesh)
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j);hx, hy = mesh.blocks[ref.block].spacing
        result += TwoPhaseAMR._get(f.fraction, ref) * hy *
            (sinpi(2 * (x + hx / 2)) - sinpi(2 * (x - hx / 2))) / pi
    end
    return result
end
function wave_case(kind, n, dr, sigma, gravity)
    mean_height = 0.43
    phi = (x, y) -> y - mean_height - 0.002cospi(2x)
    mesh = BlockTreeMesh(
        block_size = (n, n), max_level = 2, periodic = (true, false),
        refine = (k, o, e) -> kind == :uniform ? k.level == 0 :
            kind == :mixed && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
    )
    f = InterfaceField(mesh, phi)
    indicator = InterfaceRefinementIndicator(f; level = 2)
    if kind == :amr
        work = AMRWorkspace(mesh, f)
        for step in 1:2
            adapt!(work, (indicator,); step, force = true)
            initialize!(f, phi) # setup only: initialize exact geometry on chosen leaves
        end
    end
    flow = FlowSimulation(
        mesh; interface = f, viscosity = (0.0, 0.0), density = (dr, 1.0),
        surface_tension = sigma, gravity = (0.0, -gravity),
        refinement_indicators = kind == :amr ? (indicator,) : ()
    )
    k = 2pi
    omega = sqrt(((dr - 1) * gravity * k + sigma * k^3) / (dr * coth(k * mean_height) + coth(k * (1 - mean_height))))
    period = omega == 0 ? 0.5 : 2pi / omega
    a0 = wave_amplitude(f);mass = integrate(f);errors = Float64[]
    for target in range(period / 4, period; length = 4)
        dt = min(period / 80, 0.8stable_timestep(flow))
        simulate!(flow, target; dt, rtol = 1.0e-10, atol = 1.0e-12)
        measured = wave_amplitude(f) / a0
        push!(errors, abs(measured - cos(omega * flow.time)))
        @test integrate(f) ≈ mass atol = 2.0e-10
        @test isfinite(measured)
    end
    error = sqrt(sum(abs2, errors) / length(errors))
    println("wave mesh=$kind n=$n DR=$dr sigma=$sigma gravity=$gravity normalized_amplitude_L2_error=$error")
    return error
end
if abspath(PROGRAM_FILE) == @__FILE__
    @testset "Capillary and gravity waves against finite-depth linear theory" begin
        for kind in (:uniform, :mixed, :amr),dr in (1.0, 1000.0),
                (sigma, gravity) in (((dr + 1) / 2, 0.0), (0.0, 1.0), ((dr + 1) / 2, 1.0))
            resolutions = kind == :uniform ? (8, 16, 32) : (4, 8, 16)
            errors = [wave_case(kind, n, dr, sigma, gravity) for n in resolutions]
            rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
            println("wave convergence mesh=$kind DR=$dr sigma=$sigma gravity=$gravity n=$resolutions errors=$errors rates=$rates")
            @test all(diff(errors) .< 0)
            @test all(rates .>= 1.5)
        end
    end
end
