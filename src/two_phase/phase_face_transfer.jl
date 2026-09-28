"""Register intrinsic staggered phase scalars for accurate face transfer."""

function register_adaptive_field!(workspace::AMRWorkspace, q::PhaseFaceScalar)
    register_adaptive_field!(workspace, q.interface)
    index = _find_field_buffer(workspace, q.interface.fraction)
    strategy = workspace.field_buffers[index].prolongation
    any(s -> s === q, strategy.staggered) && return workspace
    _phase_face_average!(
        q.fraction, q.interface.conditional_fraction, q.interface.geometry,
    )
    push!(strategy.staggered, q)
    # Intrinsic values use limited linear face interpolation. A global face
    # integral correction is inappropriate because this is not a stored mass.
    for field in (q.liquid, q.gas)
        register_adaptive_field!(
            workspace, field;
            piecewise_constant = false, conserve_integral = false
        )
    end
    return workspace
end
function _finish_phase_face_transfer!(q::PhaseFaceScalar{Axis, T}) where {Axis, T}
    mesh = q.interface.mesh
    _resize_face_field!(q.fraction, mesh)
    count = length(q.liquid.interior) + length(q.liquid.boundary)
    for values in (
            q.scratch.source_liquid, q.scratch.source_gas,
            q.scratch.liquid_fraction, q.scratch.predicted_liquid,
            q.scratch.gas_fraction, q.scratch.predicted_gas,
            q.scratch.values, q.scratch.increment, q.scratch.rates,
        )
        resize!(values, count)
        fill!(values, zero(T))
    end
    _phase_face_average!(
        q.fraction, q.interface.conditional_fraction,
        q.interface.geometry,
    )
    _resize_zero_face_field!(q.workspace.tendency)
    _resize_zero_face_field!(q.workspace.rates)
    work = _phase_advection_workspace(q.workspace.tendency, q.workspace.rates,
        q.interface.geometry, Ref(true); reuse_flux = q.workspace.dual_flux)
    q.workspace = work
    return q
end
