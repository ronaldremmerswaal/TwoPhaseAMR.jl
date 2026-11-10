# Shared prelude of the audit of ch:overview: loads an INSTRUMENTED copy of the package, `TracedAMR`.
#
# The copy is built from the unmodified sources of src/ by `include(mapexpr, ...)`: every top-level function definition whose name is
# in `TRACED` gets one extra first statement that appends (name, file, selected argument values) to `Tracer.LOG`; nothing else
# changes. The external dependencies (Krylov, ...) are those of the real package, loaded by `using TwoPhaseAMR`.
#
# Use: include(".../_trace.jl"); using .Tracer; A = Tracer.AMR  (the instrumented module);  Tracer.record() do ... end.
# Run (only as part of a script): julia --heap-size-hint=3G -t 1 --project=docs/techref/figures <script>

module Tracer

using TwoPhaseAMR  # loads the dependencies; the instrumented copy below is a separate module

const LOG = Any[]
const ENABLED = Ref(false)

function note!(name::Symbol, file::String, names::Tuple, values::Tuple)
    ENABLED[] && push!(LOG, (name = name, file = file, args = NamedTuple{names}(values)))
    return nothing
end

"""Record `f()` and return `(result, log)`; the log of an exception is kept in `LOG`."""
function record(f)
    empty!(LOG)
    ENABLED[] = true
    try
        return f(), copy(LOG)
    finally
        ENABLED[] = false
    end
end

# function name => argument names whose values are recorded (positional or keyword names of the signature)
const TRACED = Dict{Symbol, Vector{Symbol}}(
    :step! => [:dt, :time, :_reference_dt],
    :_enforce_current_velocity_boundary! => [:time, :preserve_open],
    :_enforce_phase_velocity_boundary! => [:time],
    :_enforce_predictor_velocity_boundary! => [:time, :enforce_geometry],
    :enforce_velocity_boundary! => [:time, :preserve_open],
    :enforce_geometry_no_slip! => Symbol[],
    :_prepare_step_controls! => Symbol[],
    :_check_transport_timestep => [:dt],
    :_transport_substep! => [:dt, :time, :geometry_prepared],
    :_body_force_substep! => [:midpoint],
    :_diffusion_substep! => [:time],
    :_refresh_pressure_substep! => Symbol[],
    :_one_velocity_pressure_solver! => [:refresh],
    :_phase_force_substep! => Symbol[],
    :_apply_phase_forces! => Symbol[],
    :surface_tension! => Symbol[],
    :gravity! => Symbol[],
    :_pressure_predictor! => Symbol[],
    :_project_prepared! => [:dt, :_reference_dt],
    :_project_two_velocity! => [:pressure_prepared, :_reference_dt],
    :_capture_two_velocity_inherited_flux! => Symbol[],
    :_two_velocity_pressure_operator! => Symbol[],
    :_phase_force_pressure_operator! => Symbol[],
    :reconstruct! => Symbol[],
    :implicit_midpoint_diffusion! => Symbol[],
    :_advect_wy! => [:sweep_order, :initial_geometry_prepared, :check_timestep],
    :_compute_directional_wy_fluxes! => [:axis],
    :_dispatch_convection! => [:sweep_axis],
    :adapt! => [:step, :time],
    :_rebuild_topology_dependent_state! => Symbol[],
    :_project_regridded_two_velocity! => [:dt, :time],
    :_record_diagnostics! => [:event],
    :_observe_safely! => [:event],
    :_observer_stop_requested => Symbol[],
    :_prepare_independent_step! => Symbol[],
    :stable_timestep => Symbol[],
    :_projection_weighted_gradient! => Symbol[],
    :_add_common_phase_acceleration! => [:dt],
    :_navier_stokes_amr_workspace => Symbol[],
    :_refresh_geometric_multigrid! => Symbol[],
)

const EXTERNAL = Dict(
    :RegionTrees => Base.PkgId(Base.UUID("dee08c22-ab7f-5625-9660-a9af2021b33f"), "RegionTrees"),
    :Krylov => Base.PkgId(Base.UUID("ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"), "Krylov"),
    :Polyester => Base.PkgId(Base.UUID("f517fe37-dbe3-4b94-8317-1923a5111588"), "Polyester"),
    :IncompleteLU => Base.PkgId(Base.UUID("40713840-3770-5561-ab4c-a76e7d0d7895"), "IncompleteLU"),
    :StaticArrays => Base.PkgId(Base.UUID("90137ffa-7385-5640-81b9-e52037218182"), "StaticArrays"),
    :PrecompileTools => Base.PkgId(Base.UUID("aea7be01-6a6a-4083-8856-8a6e6704d82a"), "PrecompileTools"),
)

function _argname(a)
    a isa Symbol && return a
    a isa Expr || return nothing
    (a.head === :(::) && length(a.args) == 2) && return _argname(a.args[1])
    (a.head === :kw || a.head === :(=)) && return _argname(a.args[1])
    return nothing
end

# (name, names of all positional and keyword arguments) of a method signature, or nothing
function _callinfo(sig)
    while sig isa Expr && sig.head in (:where, :(::))
        sig = sig.args[1]
    end
    (sig isa Expr && sig.head === :call && sig.args[1] isa Symbol) || return nothing
    names = Symbol[]
    for a in sig.args[2:end]
        if a isa Expr && a.head === :parameters
            for k in a.args
                n = _argname(k)
                n === nothing || push!(names, n)
            end
        else
            n = _argname(a)
            n === nothing || push!(names, n)
        end
    end
    return (name = sig.args[1], args = names)
end

function instrument(ex, file::String)
    ex isa Expr || return ex
    if ex.head === :call && length(ex.args) == 2 && ex.args[1] === :include && ex.args[2] isa String
        path = ex.args[2]
        return Expr(:call, :include, Expr(:->, :e, Expr(:call, instrument, :e, path)), path)
    end
    if (ex.head === :import || ex.head === :using)
        out = deepcopy(ex)
        for clause in out.args
            path = clause.head === :(:) ? clause.args[1] : clause
            if path.head === :. && path.args[1] in keys(EXTERNAL)
                pushfirst!(path.args, :Main)
            end
        end
        return out
    end
    if ex.head === :module && ex.args[2] === :TwoPhaseAMR
        return Expr(:module, ex.args[1], :TracedAMR, instrument(ex.args[3], file))
    end
    if ex.head === :function || (ex.head === :(=) && ex.args[1] isa Expr && _callinfo(ex.args[1]) !== nothing)
        info = _callinfo(ex.args[1])
        if info !== nothing && haskey(TRACED, info.name) && length(ex.args) == 2
            recorded = filter(in(info.args), TRACED[info.name])
            note = Expr(
                :call, note!, QuoteNode(info.name), file, Expr(:tuple, QuoteNode.(recorded)...),
                Expr(:tuple, recorded...),
            )
            body = ex.args[2]
            newbody = body isa Expr && body.head === :block ? Expr(:block, note, body.args...) : Expr(:block, note, body)
            return Expr(ex.head, ex.args[1], newbody)
        end
        return ex
    end
    return Expr(ex.head, (instrument(a, file) for a in ex.args)...)
end

function load_instrumented(root::String)
    for (name, id) in EXTERNAL
        Core.eval(Main, :(const $name = $(Base.loaded_modules[id])))
    end
    Base.include(e -> instrument(e, "TwoPhaseAMR.jl"), Main, joinpath(root, "src", "TwoPhaseAMR.jl"))
    return getfield(Main, :TracedAMR)
end

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const AMR = Base.invokelatest(getfield, (load_instrumented(ROOT); Main), :TracedAMR)

"""Names (and recorded arguments) of the log entries restricted to `names`."""
select(log, names) = [e for e in log if e.name in names]

end # module
