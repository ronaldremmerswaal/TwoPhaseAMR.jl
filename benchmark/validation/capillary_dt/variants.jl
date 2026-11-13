# In-script variant: after the projection, remove the phase-velocity jump on a
# selected class of faces by setting u_l = u_g = (a_l u_l + a_g u_g)/(a_l + a_g),
# the aperture-weighted face value. This keeps every cell's mixture flux, so the
# projected mixture divergence is unchanged and the VOF transport stays
# conservative. src/ is not modified.

"""
Faces of the last projection: `:all`; `:rowless`, mixed faces without a retained
jump row (unresolved stencils are dropped, cmdgfm_projection.jl:914-921);
`:infer`, rows that take the inference branch (cmdgfm_projection.jl:656-708).
"""
function face_class(flow, class)
    model = flow.system.two_phase
    state = TP._two_velocity_flow_workspace!(flow.workspace.two_phase, model)
    work = state.projection_workspace
    data = work.pressure_cache.face_data
    class === :all && return collect(eachindex(data.liquid_fraction)), data
    with_row = Set(s.face for s in work.stencils)
    class === :rowless && return [i for i in data.mixed if !(i in with_row)], data
    class === :infer && return [s.face for s in work.stencils if s.gas_inference_face > 0], data
    throw(ArgumentError("unknown face class $class"))
end

function equalize_hook(class)
    return (velocity, flow) -> begin
        faces, data = face_class(flow, class)
        for i in faces
            l, g = velocity[1].interior[i], velocity[2].interior[i]
            (isfinite(l) && isfinite(g)) || continue
            al, ag = data.liquid_aperture[i], data.gas_aperture[i]
            u = al + ag > 0 ? (al * l + ag * g) / (al + ag) :
                (data.liquid_mass[i] * l + data.gas_mass[i] * g) / data.total_mass[i]
            velocity[1].interior[i] = u
            velocity[2].interior[i] = u
        end
        return velocity
    end
end
