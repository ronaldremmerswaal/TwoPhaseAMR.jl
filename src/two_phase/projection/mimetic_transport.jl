# STUDY PROTOTYPE (branch study/mm-momentum-transport), not a default path.
#
# Metric-consistent completion of the two-velocity momentum transport for
# `MimeticProjection`. The phase values are first transported as before, in
# the lumped metric `W` (diagonal staggered phase masses, VOF volume fluxes,
# directional Lax--Wendroff sweeps); this file then maps the result into the
# Gram metric `𝓜` of the projection with one solve of its band block:
#
#   𝓜⁺ u⁺ = 𝓜⁺ ū + W⁺(u_W − ū) + (1 − θ)(c − E⁺ ū) [+ Y ū + Ψ(ū)]
#
# with `E = 𝓜 − W`, `ū` the extended start values, `u_W` the values after the
# `W` sweeps, `c = Eⁿuⁿ` mapped to the new degrees of freedom, and for the
# redistributed design (`:redistribute`, θ = 1/2) the skew band coupling `Y`
# with `Y Z = (E⁺ − Eⁿ)Z / 2` plus the remainder `Ψ` that keeps constants
# exact. See /home/ronald/Documents/mimetic-projection-audit/notes/
# mimetic_momentum_transport.tex.
#
# `_MIMETIC_TRANSPORT[] = :none` (default) leaves the step untouched unless a
# diagnostics log is installed in `_MIMETIC_TRANSPORT_LOG[]`.

import Krylov

const _MIMETIC_TRANSPORT = Ref{Symbol}(:none)
const _MIMETIC_TRANSPORT_THETA = Ref(1.0)
const _MIMETIC_TRANSPORT_LOG = Ref{Any}(nothing)
"""Apply the momentum restoration of `:increment_conservative` after any design."""
const _MIMETIC_TRANSPORT_FIX = Ref(false)

struct _MimeticTransportSnapshot{T}
    dof_face::Vector{Int}
    dof_phase::Vector{Int8}
    dof_liquid_weight::Vector{T}
    excess::Vector{T}
    excess_constant::Matrix{T}
    momentum::NTuple{6, T}
    energy::T
    w_momentum::NTuple{2, T}
    excess_columns::NTuple{6, T}
    excess_tensor::Matrix{T}
    mass_tensor::Matrix{T}
    assembly_time::Float64
end

# Columns of `Z`: (liquid x, liquid y, gas x, gas y); an identified grid-line
# value belongs to both phases.
@inline function _mimetic_constant_column(axis, phase, column)
    column_phase = column <= 2 ? 1 : 2
    column_axis = isodd(column) ? 1 : 2
    return axis == column_axis && (phase == column_phase || phase == 3)
end

function _mimetic_constant_fields(assembly, mesh)
    T = eltype(assembly.dof_mass)
    dofs = length(assembly.dof_face)
    Z = zeros(T, dofs, 4)
    for dof in 1:dofs
        axis = mesh.faces[assembly.dof_face[dof]].axis
        for column in 1:4
            _mimetic_constant_column(axis, assembly.dof_phase[dof], column) &&
                (Z[dof, column] = one(T))
        end
    end
    return Z
end

# Momentum `Zᵀ𝓜u` per column of `Z`, then the mixture x and y momentum (an
# identified grid-line value counted once).
function _mimetic_columns(Z, v, assembly, mesh)
    T = eltype(v)
    mixture = zeros(T, 2)
    for dof in eachindex(v)
        mixture[mesh.faces[assembly.dof_face[dof]].axis] += v[dof]
    end
    return (ntuple(c -> dot(view(Z, :, c), v), 4)..., mixture[1], mixture[2])
end

function _mimetic_w_momentum(assembly, mesh, values)
    T = eltype(values)
    p = zeros(T, 2)
    for dof in eachindex(values)
        p[mesh.faces[assembly.dof_face[dof]].axis] += assembly.dof_mass[dof] * values[dof]
    end
    return (p[1], p[2])
end

_mimetic_transport_active() =
    _MIMETIC_TRANSPORT[] !== :none || !isnothing(_MIMETIC_TRANSPORT_LOG[])

_mimetic_transport_before!(state, velocity) = nothing
function _mimetic_transport_before!(
        state::TwoVelocityFlowWorkspace{<:Any, <:Any, <:Any, <:Any, <:MimeticProjectionSolver},
        velocity,
    )
    _mimetic_transport_active() || return nothing
    solver = state.projection_solver
    work = state.projection_workspace
    mesh = solver.interface.mesh
    assembly_time = @elapsed assembly = _mimetic_assemble!(solver, work)
    values = eltype(assembly.dof_mass)[]
    _mimetic_gather!(values, assembly, velocity)
    metric = _mimetic_metric_matrix(work)
    mu = metric * values
    wn = _PHASE_ACTIVITY[] === :aperture ?
        first(_mimetic_merged_masses(assembly, work.face_data, metric, work.topology, mesh)) :
        assembly.dof_mass
    excess = mu .- wn .* values
    Z = _mimetic_constant_fields(assembly, mesh)
    excess_constant = metric * Z .- wn .* Z
    return _MimeticTransportSnapshot(
        copy(assembly.dof_face), copy(assembly.dof_phase),
        copy(assembly.dof_liquid_weight), excess, excess_constant,
        _mimetic_columns(Z, mu, assembly, mesh), dot(values, mu) / 2,
        _mimetic_w_momentum(assembly, mesh, values),
        _mimetic_columns(Z, excess, assembly, mesh), Z' * excess_constant, Z' * (metric * Z),
        assembly_time,
    )
end

# Map a covector of the old degrees of freedom to the new ones; a grid-line
# value is split between the phases by its liquid mass weight. Returns the
# amount (per column of `Z`) that has no new degree of freedom.
function _mimetic_map_covector!(destination, snapshot, assembly, source, mesh)
    T = eltype(destination)
    orphan = zeros(T, 4)
    fill!(destination, zero(T))
    for dof in eachindex(source)
        value = source[dof]
        iszero(value) && continue
        face = snapshot.dof_face[dof]
        phase = snapshot.dof_phase[dof]
        axis = mesh.faces[face].axis
        parts = phase == 3 ?
            ((1, snapshot.dof_liquid_weight[dof]), (2, one(T) - snapshot.dof_liquid_weight[dof])) :
            ((Int(phase), one(T)),)
        for (target_phase, weight) in parts
            target = target_phase == 1 ? assembly.liquid_dof[face] : assembly.gas_dof[face]
            if target > 0
                destination[target] += weight * value
            else
                orphan[(target_phase - 1) * 2 + axis] += weight * value
            end
        end
    end
    return orphan
end

# Face value of `velocity` on every degree of freedom (mass-weighted mean on
# an identified grid-line value); `fallback` where it is not finite.
function _mimetic_dof_values!(values, assembly, liquid, gas, fallback = nothing)
    T = eltype(values)
    for dof in eachindex(values)
        face = assembly.dof_face[dof]
        phase = assembly.dof_phase[dof]
        value = if phase == 3
            w = assembly.dof_liquid_weight[dof]
            w * liquid[face] + (one(T) - w) * gas[face]
        else
            phase == 1 ? liquid[face] : gas[face]
        end
        if !isfinite(value)
            isnothing(fallback) && throw(ArgumentError("non-finite phase value on face $face"))
            value = fallback[dof]
        end
        values[dof] = value
    end
    return values
end

# Extended start values of the transport (the frozen sources of its sweeps),
# as phase arrays indexed by mesh face.
function _mimetic_transport_sources(state, mesh)
    T = eltype(state.convected[1].interior)
    faces = length(mesh.faces)
    liquid = fill(T(NaN), faces)
    gas = fill(T(NaN), faces)
    counters = (Ref(0), Ref(0))
    for (index, face) in pairs(mesh.faces)
        counter = counters[face.axis]
        counter[] += 1
        q = face.axis == 1 ? state.momentum.x : state.momentum.y
        liquid[index] = q.scratch.source_liquid[counter[]]
        gas[index] = q.scratch.source_gas[counter[]]
    end
    return liquid, gas
end

"""STUDY: largest staggered mass a value keeps under aperture activity, in units
of its metric inertia; the rest is merged into a donor."""
const _MIMETIC_MERGE_KAPPA = Ref(1.0)

"""Transported phase values after the `W` sweeps (before the final marking), as
phase arrays indexed by mesh face."""
function _mimetic_transport_raw(state, mesh)
    T = eltype(state.convected[1].interior)
    liquid = fill(T(NaN), length(mesh.faces))
    gas = fill(T(NaN), length(mesh.faces))
    counters = (Ref(0), Ref(0))
    for (index, face) in pairs(mesh.faces)
        counter = counters[face.axis]
        counter[] += 1
        q = face.axis == 1 ? state.momentum.x : state.momentum.y
        liquid[index] = q.liquid.interior[counter[]]
        gas[index] = q.gas.interior[counter[]]
    end
    return liquid, gas
end

"""STUDY (aperture activity): merged staggered phase masses `W̃` on the degrees
of freedom. A phase value keeps at most `κ 𝓜_ff` of its staggered mass `m_f`;
the excess, and the whole mass of a value without degree of freedom (phase
volume, no wetted face), goes to the same-phase value of the face ring with
the largest aperture (same axis: the constant fields and the totals `ZᵀW̃Z =
ZᵀWZ` are kept). With `increment` (per phase, per face) the same merge maps
the covector `m_f Δu_f`."""
function _mimetic_merged_masses(assembly, data, metric, topology, mesh; increment = nothing)
    T = eltype(assembly.dof_mass)
    kappa = T(_MIMETIC_MERGE_KAPPA[])
    dofs = length(assembly.dof_face)
    merged = zeros(T, dofs)
    covector = zeros(T, dofs)
    moved = 0
    dropped = zero(T)
    delta(phase, face) = isnothing(increment) ? zero(T) :
        (d = increment[phase][face]; isfinite(d) ? d : zero(T))
    for face in eachindex(mesh.faces)
        volume = mesh.face_dual_volumes[face]
        identified = assembly.liquid_dof[face]
        if identified > 0 && identified == assembly.gas_dof[face]
            for phase in 1:2
                m = volume * (phase == 1 ? data.liquid_mass[face] : data.gas_mass[face])
                merged[identified] += m
                covector[identified] += m * delta(phase, face)
            end
            continue
        end
        for phase in 1:2
            m = volume * (phase == 1 ? data.liquid_mass[face] : data.gas_mass[face])
            m > zero(T) || continue
            dof_at(g) = g <= 0 ? 0 : (phase == 1 ? assembly.liquid_dof[g] : assembly.gas_dof[g])
            dof = dof_at(face)
            keep = dof > 0 ? min(m, kappa * metric[dof, dof]) : zero(T)
            excess = m - keep
            d = delta(phase, face)
            if dof > 0
                merged[dof] += keep
                covector[dof] += keep * d
            end
            excess > zero(T) || continue
            ring = _mimetic_face_ring(topology, face, mesh.faces[face].axis)
            donor = _mimetic_donor(
                topology, mesh, ring, dof_at, g -> (e = dof_at(g); e > 0 && e != dof),
                assembly.dof_aperture,
            )
            target = donor > 0 ? donor : dof
            if target > 0
                merged[target] += excess
                covector[target] += excess * d
                moved += donor > 0
            else
                dropped += excess
            end
        end
    end
    return merged, covector, (; moved, dropped)
end

"""Skew band coupling `Y` (edge values `y` on same-phase pairs of faces of a
cell) with `Y Z ≈ d` in the least-squares sense, minimal norm. Returns the
edges, `y`, the residual `d - Y Z` and the LSQR iteration count."""
function _mimetic_redistribution(assembly, topology, mesh, d, metric)
    T = eltype(d)
    dofs = size(d, 1)
    scale = maximum(abs, d; init = zero(T))
    active = falses(dofs)
    for dof in 1:dofs
        any(c -> abs(d[dof, c]) > T(1.0e-14) * scale, 1:4) && (active[dof] = true)
    end
    cells = size(topology.sides, 2)
    dof_at(face, phase) = face <= 0 ? 0 :
        (phase == 1 ? assembly.liquid_dof[face] : assembly.gas_dof[face])
    # One ring: every value of a cell that holds an active value.
    grow = copy(active)
    for cell in 1:cells, phase in 1:2
        any(k -> (d0 = dof_at(topology.sides[k, cell], phase); d0 > 0 && active[d0]), 1:4) || continue
        for k in 1:4
            d0 = dof_at(topology.sides[k, cell], phase)
            d0 > 0 && (grow[d0] = true)
        end
    end
    edges = Tuple{Int, Int}[]
    add_edge(f, g) = (f > 0 && g > 0 && f != g && grow[f] && grow[g]) &&
        push!(edges, (min(f, g), max(f, g)))
    for cell in 1:cells, phase in 1:2, k in 1:4, l in (k + 1):4
        add_edge(dof_at(topology.sides[k, cell], phase), dof_at(topology.sides[l, cell], phase))
    end
    # Couplings of the metric (Gram blocks, ties) and the transverse neighbours
    # of the same axis (the face ring).
    rows_m = rowvals(metric)
    for g in 1:dofs, pointer in nzrange(metric, g)
        add_edge(rows_m[pointer], g)
    end
    for dof in 1:dofs
        grow[dof] || continue
        face = assembly.dof_face[dof]
        phase = assembly.dof_phase[dof] == 2 ? 2 : 1
        for g in _mimetic_face_ring(topology, face, mesh.faces[face].axis)
            g == 0 && continue
            add_edge(dof, dof_at(g, phase))
            assembly.dof_phase[dof] == 3 && add_edge(dof, dof_at(g, 2))
        end
    end
    sort!(edges)
    unique!(edges)
    local_index = zeros(Int, dofs)
    members = findall(grow)
    for (i, dof) in pairs(members)
        local_index[dof] = i
    end
    rows = Int[]
    columns = Int[]
    values = T[]
    axis_of(dof) = mesh.faces[assembly.dof_face[dof]].axis
    for (e, (f, g)) in pairs(edges)
        for c in 1:4
            if _mimetic_constant_column(axis_of(g), assembly.dof_phase[g], c)
                push!(rows, (local_index[f] - 1) * 4 + c); push!(columns, e); push!(values, one(T))
            end
            if _mimetic_constant_column(axis_of(f), assembly.dof_phase[f], c)
                push!(rows, (local_index[g] - 1) * 4 + c); push!(columns, e); push!(values, -one(T))
            end
        end
    end
    A = sparse(rows, columns, values, 4length(members), length(edges))
    b = zeros(T, 4length(members))
    for (i, dof) in pairs(members), c in 1:4
        b[(i - 1) * 4 + c] = d[dof, c]
    end
    y, stats = Krylov.lsqr(
        A, b; atol = zero(T), btol = zero(T), etol = zero(T),
        axtol = T(1.0e-15), itmax = 20 * length(edges) + 100
    )
    residual = zeros(T, dofs, 4)
    fitted = A * y
    for (i, dof) in pairs(members), c in 1:4
        residual[dof, c] = d[dof, c] - fitted[(i - 1) * 4 + c]
    end
    # Outside the active ring `d` vanishes and so does the fit.
    for dof in 1:dofs
        grow[dof] && continue
        for c in 1:4
            residual[dof, c] = d[dof, c]
        end
    end
    return edges, y, residual, stats.niter, members
end

_mimetic_transport_after!(::Nothing, state, model) = nothing
function _mimetic_transport_after!(snapshot::_MimeticTransportSnapshot{T}, state, model) where {T}
    mode = _MIMETIC_TRANSPORT[]
    solver = state.projection_solver
    work = state.projection_workspace
    mesh = solver.interface.mesh
    assembly_time = @elapsed assembly = _mimetic_assemble!(solver, work)
    dofs = length(assembly.dof_face)
    liquid_old, gas_old = _mimetic_transport_sources(state, mesh)
    old = _mimetic_dof_values!(zeros(T, dofs), assembly, liquid_old, gas_old)
    transported = _mimetic_dof_values!(
        zeros(T, dofs), assembly, state.convected[1].interior, state.convected[2].interior, old,
    )
    metric = _mimetic_metric_matrix(work)
    Z = _mimetic_constant_fields(assembly, mesh)
    merge_stats = (moved = 0, dropped = zero(T))
    if _PHASE_ACTIVITY[] === :aperture
        raw_liquid, raw_gas = _mimetic_transport_raw(state, mesh)
        increment = (raw_liquid .- liquid_old, raw_gas .- gas_old)
        mass, rhs, merge_stats = _mimetic_merged_masses(
            assembly, work.face_data, metric, work.topology, mesh; increment,
        )
    else
        mass = assembly.dof_mass
        rhs = mass .* (transported .- old)
    end
    theta = mode === :theta ? T(_MIMETIC_TRANSPORT_THETA[]) :
        mode === :redistribute ? T(1 // 2) : one(T)
    orphan = zeros(T, 4)
    redistribution = (iterations = 0, residual = zero(T), obstruction = zeros(T, 4), time = 0.0, edges = 0)
    if mode in (:theta, :redistribute) && theta != one(T)
        mapped = zeros(T, dofs)
        orphan = _mimetic_map_covector!(mapped, snapshot, assembly, snapshot.excess, mesh)
        excess_new = metric * old .- mass .* old
        rhs .+= (one(T) - theta) .* (mapped .- excess_new)
        if mode === :redistribute
            rtime = @elapsed begin
                mapped_constant = zeros(T, dofs, 4)
                for c in 1:4
                    _mimetic_map_covector!(
                        view(mapped_constant, :, c), snapshot, assembly,
                        view(snapshot.excess_constant, :, c), mesh,
                    )
                end
                d = ((metric * Z .- mass .* Z) .- mapped_constant) ./ 2
                edges, y, residual, iterations, members = _mimetic_redistribution(
                    assembly, work.topology, mesh, d, metric,
                )
                for (e, (f, g)) in pairs(edges)
                    rhs[f] += y[e] * old[g]
                    rhs[g] -= y[e] * old[f]
                end
                # Remainder: constants stay exact with the mass-weighted mean
                # of each column over the ring.
                for c in 1:4
                    total = weight = zero(T)
                    for dof in members
                        Z[dof, c] > 0 || continue
                        total += mass[dof] * old[dof]
                        weight += mass[dof]
                    end
                    mean = weight > 0 ? total / weight : zero(T)
                    for dof in 1:dofs
                        rhs[dof] += residual[dof, c] * mean
                    end
                end
            end
            obstruction = [sum(view(residual, :, c)) for c in 1:4]
            redistribution = (
                iterations, residual = maximum(abs, residual; init = zero(T)),
                obstruction, time = rtime, edges = length(edges),
            )
        end
    end
    fixer = zeros(T, 4)
    if mode === :increment_conservative || _MIMETIC_TRANSPORT_FIX[]
        # Restore the 𝓜-momentum: the update gives 𝓜⁺u⁺ = 𝓜⁺ū + rhs, so its
        # change is Zᵀ(𝓜⁺ū + rhs) - Zᵀ𝓜ⁿuⁿ; less the constant part
        # Zᵀ(𝓜⁺ - 𝓜ⁿ)Z⟨ū⟩ (the Gram-versus-VOF mass obstruction, zero with
        # mass-matched points) it is removed by a mass-proportional uniform
        # force per phase and axis. A design that keeps constants exactly
        # gives a zero force for a constant field.
        predicted = Z' * (metric * old .+ rhs)
        MT = Z' * (metric * Z)
        means = zeros(T, 4)
        totals = zeros(T, 4)
        for c in 1:4
            weighted = zero(T)
            for dof in 1:dofs
                (Z[dof, c] > 0 && assembly.dof_phase[dof] != 3) || continue
                weighted += mass[dof] * old[dof]
                totals[c] += mass[dof]
            end
            means[c] = totals[c] > 0 ? weighted / totals[c] : zero(T)
        end
        for c in 1:4
            totals[c] > 0 || continue
            defect = predicted[c] - snapshot.momentum[c]
            for k in 1:4
                defect -= (MT[c, k] - snapshot.mass_tensor[c, k]) * means[k]
            end
            fixer[c] = defect
            for dof in 1:dofs
                (Z[dof, c] > 0 && assembly.dof_phase[dof] != 3) || continue
                rhs[dof] -= mass[dof] / totals[c] * defect
            end
        end
    end
    log = _MIMETIC_TRANSPORT_LOG[]
    w_transport = _mimetic_w_momentum(assembly, mesh, transported)
    lumped = copy(transported)
    solve_time = 0.0
    if mode !== :none
        increment = zeros(T, dofs)
        solve_time = @elapsed _mimetic_metric_solve!(increment, work, rhs)
        for dof in 1:dofs
            _mimetic_scatter!(state.convected, assembly, dof, old[dof] + increment[dof], false)
        end
        transported = _mimetic_dof_values!(
            zeros(T, dofs), assembly, state.convected[1].interior, state.convected[2].interior,
        )
    end
    if !isnothing(log)
        mu = metric * transported
        if all(iszero, orphan)
            orphan = _mimetic_map_covector!(zeros(T, dofs), snapshot, assembly, snapshot.excess, mesh)
        end
        push!(
            log, (;
                mode, theta,
                momentum_before = snapshot.momentum,
                momentum_after = _mimetic_columns(Z, mu, assembly, mesh),
                energy_before = snapshot.energy, energy_after = dot(transported, mu) / 2,
                w_momentum_before = snapshot.w_momentum,
                w_momentum_transport = w_transport,
                w_momentum_after = _mimetic_w_momentum(assembly, mesh, transported),
                excess_before = snapshot.excess_columns,
                excess_after = _mimetic_columns(Z, metric * old .- mass .* old, assembly, mesh),
                correction = maximum(abs, transported .- lumped; init = zero(T)),
                worst = let k = argmax(abs.(transported .- lumped))
                    face = assembly.dof_face[k]
                    (;
                        face, phase = assembly.dof_phase[k], aperture = assembly.dof_aperture[k],
                        mass = mass[k], metric = metric[k, k], rhs = rhs[k], old = old[k],
                        transported = lumped[k], new = transported[k],
                        was_dof = any(i -> snapshot.dof_face[i] == face, eachindex(snapshot.dof_face)),
                        band = assembly.band_position[k] > 0,
                    )
                end,
                orphan = Tuple(orphan), redistribution, fixer = Tuple(fixer), merge = merge_stats,
                mass_match = _MIMETIC_MASS_MATCH[] ? _MIMETIC_MASS_MATCH_LAST[] : nothing,
                band = length(assembly.band), dofs,
                assembly_time = snapshot.assembly_time + assembly_time, solve_time,
            )
        )
    end
    return nothing
end
