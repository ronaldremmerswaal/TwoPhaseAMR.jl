# Reproduce the inactive-liquid vorticity/coarsening defect on Stokes3.
# julia --project=. development/stokes3_performance/phase_velocity_amr.jl [0.5]
using TwoPhaseAMR, Printf
const PhaseAMR = TwoPhaseAMR
include(joinpath(@__DIR__, "..", "..", "examples", "stokes3_wave.jl"))

function phase_amr_report(flow, label)
    interface = flow.interface
    mesh = interface.mesh
    velocity = flow.velocity[1]
    criterion = VorticityRefinementIndicator(velocity;
        refine_tolerance = 0.2, coarsen_tolerance = 0.2 / 3,
        interface, phase = :liquid)
    labels = refinement_labels!(fill(:keep, nblocks(mesh)), criterion; time = flow.time)
    curl = visualization_values(velocity, Vorticity(); interface, phase = :liquid)
    inactive = invalid_active = invalid_inactive = 0
    for (phase, name) in ((1, :liquid), (2, :gas))
        for (index, face) in pairs(mesh.faces)
            active = PhaseAMR._phase_face_is_active(interface, index, face, name)
            value = flow.velocity[phase].interior[index]
            if active
                invalid_active += !isfinite(value)
            else
                inactive += 1
                invalid_inactive += !isnan(value)
            end
        end
    end
    pure_gas = blocked = 0
    nx, ny = mesh.block_size
    h = mesh.halo
    band = min(2, h)
    for block in eachindex(mesh.blocks)
        c = interface.conditional_fraction.blocks[block]
        maximum(@view c[(h+1-band):(h+nx+band), (h+1-band):(h+ny+band)]) == 0 || continue
        pure_gas += 1
        blocked += mesh.blocks[block].key.level > criterion.minimum_level &&
            labels[block] != :coarsen
        first = (block - 1) * nx * ny + 1
        all(isnan, @view curl[first:(first + nx * ny - 1)]) ||
            error("absent liquid acquired a vorticity")
    end
    @printf("PHASE_AMR %s t=%.4f faces=%d blocks=%d inactive=%d invalid_active=%d invalid_inactive=%d pure_gas_with_2cell_halo=%d blocked_coarsening=%d\n",
        label, flow.time, length(mesh.faces), nblocks(mesh), inactive,
        invalid_active, invalid_inactive, pure_gas, blocked)
    iszero(invalid_active + invalid_inactive + blocked) ||
        error("phase storage/coarsening contract failed")
    return nothing
end

function phase_amr_benchmark(final_time = 0.5)
    flow = run_example(final_time = 0.0,
        velocity_model = TwoVelocityModel(pressure_projection = CMDGFMProjection())).flow
    phase_amr_report(flow, "initial")
    measured = @timed simulate!(flow, final_time)
    phase_amr_report(flow, "late")
    @printf("ADVANCE t=%.4f wall=%.3f compile=%.3f bytes=%d steps=%d updates=%d pressure_converged=%s\n",
        final_time, measured.time, measured.compile_time, measured.bytes, measured.value.steps,
        measured.value.mesh_updates, measured.value.pressure.converged)
    return flow
end

if abspath(PROGRAM_FILE) == @__FILE__
    phase_amr_benchmark(parse(Float64, get(ARGS, 1, "0.5")))
end
