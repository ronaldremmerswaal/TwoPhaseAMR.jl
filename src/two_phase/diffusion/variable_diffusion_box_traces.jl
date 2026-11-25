# Condense the free tangential trace at each Cartesian boundary vertex. A
# regular vertex has only its own shear row, which vanishes on minimization.
# Green remainders at cuts and level transitions can share several traces:
# their connected rows must be minimized together with their actual viscosities.
struct _LSBoxPatch{T}
    rows::Vector{Int}
    faces::Vector{Int}
    basis::Matrix{T}                   # orthonormal basis for the geometric trace columns
    raw::Matrix{T}                     # strains after the cell-local PLIC elimination
    weighted_basis::Matrix{T}
    sqrt_weights::Vector{T}
    work::Vector{T}
    slots::Matrix{Int}                 # negative slots refer to the reaction matrix
    transpose_slots::Matrix{Int}
end

struct _LSBoxTraces{T}
    patches::Vector{_LSBoxPatch{T}}
    products::Vector{NTuple{3, Int}}   # stiffness factor slot, its two strain slots
end

function _ls_trace_basis!(B::Matrix{T}) where {T}
    # A PLIC or a terminating box segment can make two trace columns
    # dependent. Rank belongs to the geometry, not to the viscosity ratio.
    for j in axes(B, 2)
        b = view(B, :, j)
        b ./= norm(b)
    end
    size(B, 2) == 1 && return B
    f = qr!(B, ColumnNorm())
    R = f.R
    tol = maximum(size(B)) * eps(T) * abs(R[1, 1])
    k = count(j -> abs(R[j, j]) > tol, 1:min(size(B)...))
    return f.Q * Matrix{T}(I, size(B, 1), k)
end

# Mark connected rows of the bipartite trace/strain graph. In particular,
# ordinary box shear rows never create a boundary-length dense patch.
function _ls_box_groups(B)
    Bt = copy(transpose(B))
    visited = falses(size(B, 2))
    owner = zeros(Int, size(B, 1))
    groups = Tuple{Vector{Int}, Vector{Int}, Vector{Int}}[]
    for vertex in axes(B, 2)
        entries = nzrange(B, vertex)
        (visited[vertex] || isempty(entries)) && continue
        row = rowvals(B)[first(entries)]
        if length(entries) == 1 && length(nzrange(Bt, row)) == 1
            # Freeing this one trace cancels its only strain row exactly.
            owner[row] = -1
            continue
        end
        id = length(groups) + 1
        rows = Int[]
        vertices = [vertex]
        visited[vertex] = true
        i = 1
        while i <= length(vertices)
            for p in nzrange(B, vertices[i])
                r = rowvals(B)[p]
                iszero(owner[r]) || continue
                owner[r] = id
                push!(rows, r)
                for q in nzrange(Bt, r)
                    v = rowvals(Bt)[q]
                    visited[v] && continue
                    push!(vertices, v)
                    visited[v] = true
                end
            end
            i += 1
        end
        push!(groups, (sort!(rows), vertices, Int[]))
    end
    return owner, groups
end

function _ls_box_traces!(construction, nrows)
    cache = construction.ls_stag
    isempty(cache.box_rows) && return nothing
    B = dropzeros!(sparse(cache.box_rows, cache.box_vertices, cache.box_values, nrows, length(cache.keys)))
    owner, groups = _ls_box_groups(B)
    rows, cols, vals = construction.strain_rows, construction.strain_columns, construction.strain_values
    T = eltype(vals)
    for i in eachindex(rows)
        g = owner[rows[i]]
        g > 0 && push!(groups[g][3], i)
    end
    patches = _LSBoxPatch{T}[]
    for (stress, vertices, entries) in groups
        isempty(entries) && continue
        basis = _ls_trace_basis!(Matrix(B[stress, vertices]))
        size(basis, 2) == length(stress) && continue  # every strain in the patch is released
        faces = sort!(unique!(cols[entries]))
        raw = zeros(T, length(stress), length(faces))
        for p in entries
            raw[searchsortedfirst(stress, rows[p]), searchsortedfirst(faces, cols[p])] += vals[p]
        end
        push!(
            patches, _LSBoxPatch(
                stress, faces, basis, raw, similar(basis), Vector{T}(undef, length(stress)),
                Vector{T}(undef, length(stress)), similar(raw, Int), similar(raw, Int),
            ),
        )
    end
    # Each patch is the complete symbolic support of its projected rows;
    # zero entries today can become nonzero after a viscosity refresh.
    keep = 0
    for i in eachindex(rows)
        iszero(owner[rows[i]]) || continue
        keep += 1
        rows[keep] = rows[i]; cols[keep] = cols[i]; vals[keep] = vals[i]
    end
    resize!(rows, keep); resize!(cols, keep); resize!(vals, keep)
    for patch in patches, col in patch.faces, row in patch.rows
        push!(rows, row); push!(cols, col); push!(vals, zero(T))
    end
    return isempty(patches) ? nothing : _LSBoxTraces(patches, NTuple{3, Int}[])
end

function _ls_box_slots!(traces, strain, strain_transpose, reaction, prescribed)
    for patch in traces.patches, (j, f) in pairs(patch.faces)
        wall = isempty(prescribed) ? 0 : prescribed[f]
        for (i, r) in pairs(patch.rows)
            if iszero(wall)
                patch.slots[i, j] = _csc_stored_pointer(strain, r, f)
                patch.transpose_slots[i, j] = _csc_stored_pointer(strain_transpose, f, r)
            else
                patch.slots[i, j] = -_csc_stored_pointer(reaction, r, wall)
                patch.transpose_slots[i, j] = 0
            end
        end
    end
    return nothing
end

function _ls_box_products!(traces, stiffness, stresses, offsets)
    for patch in traces.patches, (j, f) in pairs(patch.faces), (k, g) in pairs(patch.faces)
        (patch.slots[1, j] < 0 || patch.slots[1, k] < 0) && continue
        entry = _csc_stored_pointer(stiffness, f, g)
        for (i, r) in pairs(patch.rows)
            product = _csc_stored_pointer_or_zero(stresses, offsets, r, entry)
            @assert !iszero(product)
            push!(traces.products, (product, patch.slots[i, j], patch.slots[i, k]))
        end
    end
    return nothing
end

# W^(-1/2)(I - QQ')W^(1/2)E, Q spanning W^(1/2)B. Orthogonalize
# twice (not normal equations), so large viscosity contrasts do not square
# the condition number. All work arrays and CSC slots survive VOF updates.
function _ls_update_box_strain!(traces, strain, strain_transpose, reaction, weights)
    values, tvalues, rvalues = nonzeros(strain), nonzeros(strain_transpose), nonzeros(reaction)
    for patch in traces.patches
        Q, w, work = patch.weighted_basis, patch.sqrt_weights, patch.work
        @inbounds for (i, r) in pairs(patch.rows)
            w[i] = sqrt(weights[r])
        end
        for j in axes(Q, 2)
            q = view(Q, :, j)
            @inbounds for i in eachindex(q)
                q[i] = w[i] * patch.basis[i, j]
            end
            for _ in 1:2, k in 1:(j - 1)
                previous = view(Q, :, k)
                coefficient = dot(previous, q)
                @. q -= coefficient * previous
            end
            q ./= norm(q)
        end
        for j in axes(patch.raw, 2)
            @inbounds for i in eachindex(w)
                work[i] = w[i] * patch.raw[i, j]
            end
            for k in axes(Q, 2)
                q = view(Q, :, k)
                coefficient = dot(q, work)
                @. work -= coefficient * q
            end
            @inbounds for i in eachindex(w)
                value = work[i] / w[i]
                slot = patch.slots[i, j]
                if slot > 0
                    values[slot] = value
                    tvalues[patch.transpose_slots[i, j]] = value
                else
                    rvalues[-slot] = value
                end
            end
        end
    end
    return nothing
end
