# Classification of the faces carrying the largest phase-velocity jumps after
# the last projection of a run (uses that projection's workspace).
function jump_faces(flow; count = 4, axis_filter = 1)
    model = flow.system.two_phase
    state = TP._two_velocity_flow_workspace!(flow.workspace.two_phase, model)
    work = state.projection_workspace
    data = work.pressure_cache.face_data
    mesh = model.interface.mesh
    v = flow.velocity
    stencil_of = Dict(s.face => s for s in work.stencils)
    candidates = [
        i for (i, f) in pairs(mesh.faces)
            if f.axis == axis_filter && isfinite(v[1].interior[i]) && isfinite(v[2].interior[i])
    ]
    sort!(candidates; by = i -> -abs(v[2].interior[i] - v[1].interior[i]))
    return map(first(candidates, count)) do i
        s = get(stencil_of, i, nothing)
        (;
            face = i, center = mesh.faces[i].center, c = data.liquid_fraction[i],
            a_l = data.liquid_aperture[i], a_g = data.gas_aperture[i],
            u_l = v[1].interior[i], u_g = v[2].interior[i],
            row = isnothing(s) ? :none : s.gas_inference_face > 0 ? :inference : :plain,
            beta = isnothing(s) ? NaN : s.beta, ratio = isnothing(s) ? NaN : s.ratio,
            donor_offsets = isnothing(s) || s.gas_inference_face == 0 ? () :
                (
                (mesh.faces[s.gas_inference_face].center[2] - mesh.faces[i].center[2]) * 16,
                (mesh.faces[s.liquid_inference_face].center[2] - mesh.faces[i].center[2]) * 16,
            ),
        )
    end
end

function print_jump_faces(emit, label, flow)
    emit("  ", label)
    emit("    ", rpad("axis face", 11), rpad("center", 18), rpad("c_face", 11), rpad("a_l", 10),
        rpad("a_g", 10), rpad("u_l", 10), rpad("u_g", 10), rpad("row", 10), rpad("beta", 9),
        rpad("n_t/n_a", 10), "donors (Δy/h, n=16)")
    for axis in (1, 2), r in jump_faces(flow; count = 3, axis_filter = axis)
        emit(
            "    ", rpad("$(axis == 1 ? "x" : "y") $(r.face)", 11),
            rpad(@sprintf("(%.4f,%.4f)", r.center...), 18), rpad(@sprintf("%.3e", r.c), 11),
            rpad(@sprintf("%.4f", r.a_l), 10), rpad(@sprintf("%.4f", r.a_g), 10),
            rpad(@sprintf("%+.4f", r.u_l), 10), rpad(@sprintf("%+.4f", r.u_g), 10),
            rpad(string(r.row), 10), rpad(@sprintf("%.4f", r.beta), 9),
            rpad(@sprintf("%.1f", r.ratio), 10), string(r.donor_offsets),
        )
    end
end
