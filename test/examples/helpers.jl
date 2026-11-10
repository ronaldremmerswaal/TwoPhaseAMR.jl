"""Load one standalone example without leaking its `run_example` method."""
function load_example(name)
    example = Module(gensym(:Example))
    examples_dir = joinpath(@__DIR__, "..", "..", "examples")
    folder = only(
        filter(readdir(examples_dir)) do entry
            endswith(entry, "_$(name)") &&
                isfile(joinpath(examples_dir, entry, "$(name).jl"))
        end
    )
    Base.include(
        example,
        joinpath(examples_dir, folder, "$(name).jl"),
    )
    return example
end

function call_example(example, name; options...)
    entrypoint = Base.invokelatest(getproperty, example, name)
    return Base.invokelatest(entrypoint; options...)
end

function example_start_probe(recorder)
    @test run_state(recorder) === :created
    @test recorder.driver isa TwoPhaseAMR.StudioDriver
    return IOBuffer()
end

function test_finite_example_fields(flow)
    mesh = flow.system.mesh
    @test all(
        isfinite(value)
            for block in eachindex(mesh.blocks)
            for value in interior(flow.pressure, block)
    )
    interface = flow.interface
    if !isnothing(interface)
        @test all(
            isfinite(value)
                for block in eachindex(mesh.blocks)
                for value in interior(interface, block)
        )
    end
    return if flow.velocity isa TwoPhaseVelocity
        for (velocity, phase) in zip(flow.velocity, (:liquid, :gas))
            # Absent phases deliberately store NaNs; active faces must be finite.
            @test all(!isinf, velocity.interior)
            @test all(!isinf, velocity.boundary)
            @test all(
                !TwoPhaseAMR._phase_face_is_active(interface, i, face, phase) ||
                    isfinite(velocity.interior[i])
                    for (i, face) in pairs(mesh.faces)
            )
            @test all(
                !TwoPhaseAMR._phase_cell_is_active(interface, face.cell, phase) ||
                    isfinite(velocity.boundary[i])
                    for (i, face) in pairs(mesh.boundary_faces)
            )
        end
    else
        @test all(isfinite, flow.velocity.interior)
        @test all(isfinite, flow.velocity.boundary)
    end
end

function test_phase_volume_conservation(flow)
    history = diagnostic_history(flow, :phase_volume)
    initial, final = first(history.values), last(history.values)
    # Each directional sweep, and each AMR transfer, may snap at most
    # fraction_floor per unit domain area during interface cleanup.
    tolerance = max(2 * flow.steps + flow.mesh_updates, 1) *
        flow.interface.fraction_floor *
        prod(flow.system.mesh.extent)
    @test all(isfinite, initial)
    @test all(isfinite, final)
    @test isapprox(final[1], initial[1]; rtol = 0, atol = tolerance)
    return @test isapprox(final[2], initial[2]; rtol = 0, atol = tolerance)
end
