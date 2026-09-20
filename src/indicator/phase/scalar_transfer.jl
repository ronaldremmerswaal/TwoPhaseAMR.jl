# Intrinsic values use limited linear prolongation. A local correction makes
# the phase-volume-weighted sum of the four fine values equal the coarse phase
# amount. Coarsening uses the corresponding phase-volume-weighted average.
struct PhaseProlongation{B, F, T}
    fraction_buffer::B
    interface::F
    liquid::Bool
    old_fraction::Dict{BlockKey, Matrix{T}}
    old_open::Dict{BlockKey, Matrix{T}}
    new_indices::Dict{BlockKey, Int}
end
function _prepare_prolongation!(p::PhaseProlongation, field, workspace)
    empty!(p.old_fraction)
    empty!(p.old_open)
    empty!(p.new_indices)
    buffer = p.fraction_buffer
    source = isempty(buffer.staging) ? buffer.field.blocks : buffer.staging
    for (b, block) in pairs(field.mesh.blocks)
        p.old_fraction[block.key] = copy(source[b])
        if !isnothing(p.interface.geometry)
            p.old_open[block.key] = copy(
                p.interface.geometry.open_fraction.blocks[b],
            )
        end
    end
    return
end
function _prepare_parallel_prolongation!(p::PhaseProlongation, workspace)
    empty!(p.new_indices)
    for (index, block) in pairs(workspace.mesh.blocks)
        p.new_indices[block.key] = index
    end
    return nothing
end
function _new_phase_fraction(p::PhaseProlongation, parent_key, fi, fj, nx, ny, halo)
    cx, ii = _fine_location(fi, nx)
    cy, jj = _fine_location(fj, ny)
    child_key = _child_keys(parent_key)[cx + 2(cy - 1)]
    block = p.new_indices[child_key]
    c = p.fraction_buffer.field.blocks[block][ii + halo, jj + halo]
    p.liquid && return c
    isnothing(p.interface.geometry) && return one(c) - c
    strategy = p.fraction_buffer.prolongation
    solid_x, solid_y, solid_intercept = strategy.geometry_planes[parent_key]
    pi, pj = (fi + 1) ÷ 2 + halo, (fj + 1) ÷ 2 + halo
    xlo = isodd(fi) ? -0.5 : 0.0
    ylo = isodd(fj) ? -0.5 : 0.0
    T = typeof(c)
    rectangle = NTuple{2, T}[
        (xlo, ylo), (xlo + 0.5, ylo),
        (xlo + 0.5, ylo + 0.5), (xlo, ylo + 0.5),
    ]
    open_polygon = _clip_polygon(
        rectangle, solid_x[pi, pj], solid_y[pi, pj],
        solid_intercept[pi, pj]; keep_less = false,
    )
    return clamp(4 * _polygon_area(open_polygon) - c, zero(T), one(T))
end
function (p::PhaseProlongation)(child, parent, halo, key)
    parent_key = _parent_key(key)
    fractions = p.old_fraction[parent_key]
    nx, ny = size(parent) .- 2halo
    for j in 1:ny,i in 1:nx
        fi = mod(key.i, 2) * nx + i
        fj = mod(key.j, 2) * ny + j
        pi = (fi + 1) ÷ 2 + halo
        pj = (fj + 1) ÷ 2 + halo
        value = parent[pi, pj]
        slope_x = _mc_slope(value - parent[pi - 1, pj], parent[pi + 1, pj] - value)
        slope_y = _mc_slope(value - parent[pi, pj - 1], parent[pi, pj + 1] - value)
        if iszero(slope_x) && iszero(slope_y)
            child[i + halo, j + halo] = value
            continue
        end
        quarter = one(value) / 4
        candidate(ii, jj) = value + (isodd(ii) ? -quarter : quarter) * slope_x +
            (isodd(jj) ? -quarter : quarter) * slope_y
        weighted = weight = zero(value)
        for fine_j in (2(pj - halo) - 1):2(pj - halo),
                fine_i in (2(pi - halo) - 1):2(pi - halo)
            w = _new_phase_fraction(p, parent_key, fine_i, fine_j, nx, ny, halo)
            weighted += w * candidate(fine_i, fine_j)
            weight += w
        end
        parent_c = fractions[pi, pj]
        parent_open = isnothing(p.interface.geometry) ? one(parent_c) :
            p.old_open[parent_key][pi, pj]
        parent_weight = p.liquid ? parent_c : parent_open - parent_c
        correction = weight > sqrt(eps(eltype(parent))) ?
            (4parent_weight * value - weighted) / weight : zero(value)
        child[i + halo, j + halo] = candidate(fi, fj) + correction
    end
    return child
end
function _restrict_parent!(p::PhaseProlongation, parent, children, halo, key)
    fraction_children = ntuple(i -> p.old_fraction[_child_keys(key)[i]], 4)
    nx, ny = size(parent) .- 2halo
    for j in 1:ny,i in 1:nx
        weighted = weight = plain = zero(eltype(parent))
        first_value = nothing
        constant = true
        for fj in (2j - 1):2j,fi in (2i - 1):2i
            cx, ii = _fine_location(fi, nx);cy, jj = _fine_location(fj, ny)
            child_index = cx + 2(cy - 1)
            value = children[child_index][ii + halo, jj + halo]
            if isnothing(first_value)
                first_value = value
            else
                constant &= value == first_value
            end
            c = fraction_children[child_index][ii + halo, jj + halo]
            child_key = _child_keys(key)[child_index]
            open = isnothing(p.interface.geometry) ? one(c) :
                p.old_open[child_key][ii + halo, jj + halo]
            w = p.liquid ? c : open - c
            weighted += w * value;weight += w;plain += value
        end
        parent[i + halo, j + halo] = constant ? first_value :
            (weight > 0 ? weighted / weight : plain / 4)
    end
    return parent
end
function register_adaptive_field!(workspace::AMRWorkspace, q::PhaseScalar{T}) where {T}
    register_adaptive_field!(workspace, q.interface)
    index = _find_field_buffer(workspace, q.interface.fraction)
    buffer = workspace.field_buffers[index]
    for (field, liquid) in ((q.liquid, true), (q.gas, false))
        strategy = PhaseProlongation(
            buffer, q.interface, liquid,
            Dict{BlockKey, Matrix{T}}(), Dict{BlockKey, Matrix{T}}(),
            Dict{BlockKey, Int}()
        )
        register_adaptive_field!(workspace, field; prolongation = strategy)
    end
    return workspace
end
