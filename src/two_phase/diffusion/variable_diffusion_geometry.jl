"""Coupled staggered `div(2μD(u))` with cell-centred normal stresses and vertex
shear stresses. The integrated stiffness is `E'WE`, so viscosity dissipates
kinetic energy for positive μ. Density enters the time-step mass matrix.

`dirichlet_faces` contains prescribed zero global face indices. Their original
physical strain columns are stored in `wall_reaction_strain`, with the same
stress rows as `strain` and columns ordered like `dirichlet_faces`; its
algebraic constraint rows are zero. For retained velocities `u`, the integrated
force on those faces is `-wall_reaction_strain' * (stress_weights .* (strain * u))`,
using the current stress weights.
"""
mutable struct VariableStaggeredDiffusionOperator{T, M, B}
    mesh::M
    stiffness::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    boundary::B
    averaging::AbstractViscosityAveraging
    strain::SparseMatrixCSC{T, Int}
    strain_transpose::SparseMatrixCSC{T, Int}
    wall_reaction_strain::SparseMatrixCSC{T, Int}
    dirichlet_faces::Vector{Int}
    stress_interpolation::SparseMatrixCSC{T, Int}
    stress_interpolation_transpose::SparseMatrixCSC{T, Int}
    stress_measures::Vector{T}
    stress_weights::Vector{T}
    cell_work::Vector{T}
    cell_inverse_work::Vector{T}
    stress_work::Vector{T}
    stiffness_pointers::Vector{Int}
    stiffness_stresses::Vector{Int}
    stiffness_factors::Vector{T}
    stiffness_refresh_offsets::Vector{Int}
end

"""One sample of velocity component `k` in the shear partition: an owned face
(`face > 0`) or a wall face normal to `k`, whose velocity is zero
(`face == 0`). Coordinates are on the doubled finest lattice, so every face
centre is an integer point: `normal` along `k`, the face span
`low:high` along the tangent `t`, and the extent `extent_low:extent_high` of
its staggered control volume along `k`. `minus` and `plus` are the adjacent
cell indices (`plus == 0` at a wall)."""
struct ShearNode
    face::Int
    normal::Int
    low::Int
    high::Int
    extent_low::Int
    extent_high::Int
    minus::Int
    plus::Int
end

"""A tangential contact of component `k` between the nodes `lower` and
`upper`, adjacent along `t` (`0` is a domain wall). Its shear region is the
rectangle `k_low:k_high` times `t_low:t_high` between the two node centres,
on the doubled lattice in the frame of `lower` (of `upper` at a lower wall);
`k_high - k_low` is the contact measure."""
struct ShearContact
    lower::Int
    upper::Int
    k_low::Int
    k_high::Int
    t_low::Int
    t_high::Int
end

"""Shear partition of one velocity component: nodes, contacts, the
tangential-derivative stencil of every contact (`stencil_faces` and
`stencil_values` in `stencil_offsets[c]:(stencil_offsets[c + 1] - 1)`) and the
contacts on the lower (`below`) and upper (`above`) side of every node."""
struct ShearPartition{T}
    nodes::Vector{ShearNode}
    order::Vector{Int}
    contacts::Vector{ShearContact}
    stencil_offsets::Vector{Int}
    stencil_faces::Vector{Int}
    stencil_values::Vector{T}
    below_offsets::Vector{Int}
    below::Vector{Int}
    above_offsets::Vector{Int}
    above::Vector{Int}
end

ShearPartition(::Type{T}) where {T} = ShearPartition{T}(
    ShearNode[], Int[], ShearContact[], Int[], Int[], T[],
    Int[], Int[], Int[], Int[],
)

"""Grow-only scratch and local topology storage for stress construction."""
mutable struct VariableDiffusionConstructionCache{T}
    cell_references::Vector{CellRef}
    strain_rows::Vector{Int}
    strain_columns::Vector{Int}
    strain_values::Vector{T}
    wall_reaction_rows::Vector{Int}
    wall_reaction_columns::Vector{Int}
    wall_reaction_values::Vector{T}
    dirichlet_columns::Vector{Int}
    interpolation_rows::Vector{Int}
    interpolation_columns::Vector{Int}
    interpolation_values::Vector{T}
    shear_partitions::NTuple{2, ShearPartition{T}}
    cell_contact_offsets::Vector{Int}
    cell_contacts::Vector{Int}
    contact_marks::Vector{Int}
    contact_candidates::Vector{Int}
    row_cells::Vector{Int}
    row_weights::Vector{T}
    stiffness_heads::Vector{Int}
    stiffness_column_counts::Vector{Int}
    stiffness_contribution_counts::Vector{Int}
    stiffness_contribution_starts::Vector{Int}
end

function VariableDiffusionConstructionCache(::Type{T}) where {T}
    return VariableDiffusionConstructionCache(
        CellRef[], Int[], Int[], T[], Int[], Int[], T[], Int[],
        Int[], Int[], T[],
        (ShearPartition(T), ShearPartition(T)),
        Int[], Int[], Int[], Int[], Int[], T[],
        Int[], Int[], Int[], Int[],
    )
end

function _cached_cell_references!(references, mesh)
    empty!(references)
    nx, ny = mesh.block_size
    sizehint!(references, ncells(mesh))
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        push!(references, CellRef(block, i, j))
    end
    return references
end

"""Columns per fixed chunk of `_symmetric_stiffness!`."""
const _STIFFNESS_COLUMN_CHUNK = 256

"""
    _symmetric_stiffness!(pointers, stresses, factors, offsets, cache, strain, strain_transpose, weights)

Return the symmetric stiffness `E'WE` for the strain `E = strain` and the
stress weights `W = Diagonal(weights)`, and fill its coefficient-refresh map.

Column `c` of the stiffness holds every face `r` that shares a stress row with
face `c`, also where the outer products cancel for the current weights:
changing the phase viscosity can make them nonzero. Its value is the mean of
the two algebraically equal halves `(A[r, c] + A[c, r]) / 2`, so it is exactly
symmetric. Each half is the left-to-right sum over the shared stresses `s` in
increasing order of `(E[s, r] W[s]) E[s, c]`, bitwise SparseArrays'
`transpose(E) * spdiagm(0 => W) * E`.

The refresh map lists, for stored entry `p`, the shared stresses `s` in the
same order in `stresses[offsets[p]:(offsets[p + 1] - 1)]` with the factors
`E[s, r] E[s, c]`; `pointers` repeats `p` once per contribution. A refresh
sums `W[s] * factor` in that order.

The columns are built by a merge of the rows of `strain_transpose` at the
stresses of each column: a count pass, a prefix sum and a fill pass, threaded
over fixed chunks of `_STIFFNESS_COLUMN_CHUNK` columns. Every column owns its
outputs, so the result is the same for every thread count.
"""
function _symmetric_stiffness!(
        pointers, stresses, factors, offsets, cache, strain, strain_transpose, weights,
    )
    T = eltype(strain)
    number_of_faces = size(strain, 2)
    strain_pointers = strain.colptr
    stress_rows = rowvals(strain)
    strain_values = nonzeros(strain)
    face_pointers = strain_transpose.colptr
    face_rows = rowvals(strain_transpose)
    transpose_values = nonzeros(strain_transpose)
    width = 0
    for column in 1:number_of_faces
        width = max(width, strain_pointers[column + 1] - strain_pointers[column])
    end
    chunks = cld(number_of_faces, _STIFFNESS_COLUMN_CHUNK)
    merge_heads = resize!(cache.stiffness_heads, chunks * width)
    column_counts = resize!(cache.stiffness_column_counts, number_of_faces)
    contribution_counts = resize!(cache.stiffness_contribution_counts, number_of_faces)
    @_parallel work = _STIFFNESS_COLUMN_CHUNK for chunk in 1:chunks
        head_offset = (chunk - 1) * width
        last_column = min(number_of_faces, chunk * _STIFFNESS_COLUMN_CHUNK)
        for column in ((chunk - 1) * _STIFFNESS_COLUMN_CHUNK + 1):last_column
            entries, contributions = _stiffness_column!(
                nothing, column, head_offset, merge_heads, strain_pointers, stress_rows,
                strain_values, face_pointers, face_rows, transpose_values, weights,
            )
            @inbounds column_counts[column] = entries
            @inbounds contribution_counts[column] = contributions
        end
    end
    column_pointers = _csr_offsets(column_counts)
    contribution_starts = _csr_offsets(
        contribution_counts, cache.stiffness_contribution_starts,
    )
    number_of_entries = column_pointers[end] - 1
    number_of_contributions = contribution_starts[end] - 1
    rows = Vector{Int}(undef, number_of_entries)
    stiffness_values = Vector{T}(undef, number_of_entries)
    resize!(pointers, number_of_contributions)
    resize!(stresses, number_of_contributions)
    resize!(factors, number_of_contributions)
    resize!(offsets, number_of_entries + 1)
    @_parallel work = _STIFFNESS_COLUMN_CHUNK for chunk in 1:chunks
        head_offset = (chunk - 1) * width
        last_column = min(number_of_faces, chunk * _STIFFNESS_COLUMN_CHUNK)
        outputs = (
            column_pointers, contribution_starts, rows, stiffness_values,
            pointers, stresses, factors, offsets,
        )
        for column in ((chunk - 1) * _STIFFNESS_COLUMN_CHUNK + 1):last_column
            _stiffness_column!(
                outputs, column, head_offset, merge_heads, strain_pointers, stress_rows,
                strain_values, face_pointers, face_rows, transpose_values, weights,
            )
        end
    end
    offsets[end] = number_of_contributions + 1
    return SparseMatrixCSC(
        number_of_faces, number_of_faces, column_pointers, rows, stiffness_values,
    )
end

"""Count (`outputs === nothing`) or fill stiffness column `column` of
`_symmetric_stiffness!`: merge the sorted face lists of the stresses of
`column`, in increasing stress order, with one head per stress in
`merge_heads[head_offset .+ (1:width)]`. Return the numbers of stored entries and
refresh contributions of the column."""
@inline function _stiffness_column!(
        outputs, column, head_offset, merge_heads, strain_pointers, stress_rows,
        strain_values, face_pointers, face_rows, transpose_values, weights,
    )
    first_stress = @inbounds strain_pointers[column]
    stress_count = (@inbounds strain_pointers[column + 1]) - first_stress
    contributions = 0
    for local_stress in 1:stress_count
        stress = @inbounds stress_rows[first_stress + local_stress - 1]
        @inbounds merge_heads[head_offset + local_stress] = face_pointers[stress]
        contributions += (@inbounds face_pointers[stress + 1]) - face_pointers[stress]
    end
    if outputs !== nothing
        column_pointers, contribution_starts, rows, stiffness_values,
            pointers, stresses, factors, offsets = outputs
        entry = @inbounds column_pointers[column]
        contribution = @inbounds contribution_starts[column]
    end
    entries = 0
    while true
        # The next face of the column: the smallest head over all stresses.
        row = typemax(Int)
        for local_stress in 1:stress_count
            stress = @inbounds stress_rows[first_stress + local_stress - 1]
            head = @inbounds merge_heads[head_offset + local_stress]
            head < (@inbounds face_pointers[stress + 1]) &&
                (row = min(row, @inbounds face_rows[head]))
        end
        row == typemax(Int) && break
        entries += 1
        if outputs !== nothing
            @inbounds offsets[entry] = contribution
            lower = zero(eltype(stiffness_values))
            upper = zero(eltype(stiffness_values))
        end
        shared = 0
        for local_stress in 1:stress_count
            stress = @inbounds stress_rows[first_stress + local_stress - 1]
            head = @inbounds merge_heads[head_offset + local_stress]
            head < (@inbounds face_pointers[stress + 1]) &&
                (@inbounds face_rows[head]) == row || continue
            @inbounds merge_heads[head_offset + local_stress] = head + 1
            outputs === nothing && continue
            shared += 1
            row_value = @inbounds transpose_values[head]
            column_value = @inbounds strain_values[first_stress + local_stress - 1]
            weight = @inbounds weights[stress]
            # SparseArrays assigns the first product and adds the others.
            lower_term = (row_value * weight) * column_value
            upper_term = (column_value * weight) * row_value
            lower = shared == 1 ? lower_term : lower + lower_term
            upper = shared == 1 ? upper_term : upper + upper_term
            @inbounds pointers[contribution] = entry
            @inbounds stresses[contribution] = stress
            @inbounds factors[contribution] = row_value * column_value
            contribution += 1
        end
        if outputs !== nothing
            @inbounds rows[entry] = row
            @inbounds stiffness_values[entry] = (lower + upper) / 2
            entry += 1
        end
    end
    return entries, contributions
end

"""
    dynamic_viscosity(interface; viscosity, averaging=HarmonicViscosityAveraging())

Return the cell-centred dynamic viscosity used by the one-velocity diffusion
operator, mixed from the phase viscosities with the conditional liquid
fraction `c` of the open fluid part of the cell and the phase-viscosity
averaging rule `averaging`.

[`HarmonicViscosityAveraging`](@ref) (the default) gives
`1 / (c / mu_liquid + (1 - c) / mu_gas)`, the series conductance across the
interface in a cell; [`GeometricViscosityAveraging`](@ref) gives the thesis'
diffuse mixture `mu_liquid^c * mu_gas^(1-c)`. The same rule averages the
viscosity of a stress row from the cell quarters it spans, see
`_fill_stress_weights!`.
"""
function dynamic_viscosity(
        field::InterfaceField{T}; viscosity::NTuple{2},
        averaging::AbstractViscosityAveraging = HarmonicViscosityAveraging(),
    ) where {T}
    phases = T.(viscosity)
    all(mu -> isfinite(mu) && mu > zero(T), phases) ||
        throw(
        ArgumentError(
            "phase dynamic viscosities must be finite and positive"
        )
    )
    result = CellField(field.mesh, T)
    mesh = field.mesh
    nx, _ = mesh.block_size
    h = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        viscosity_values = @inbounds result.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            viscosity_values[i + h, j + h] =
                _phase_dynamic_viscosity(fraction[i + h, j + h], phases, averaging)
        end
    end
    return result
end

# Signed lattice offset `delta` reduced to the nearest periodic image.
@inline _shear_wrap(delta, period, periodic) =
    periodic ? mod(delta + period ÷ 2, period) - period ÷ 2 : delta

# Length of the intersection of the intervals `a` and `b` (and `c`), summed
# over the periodic images of `b` and `c`. Frames differ by at most a period.
@inline function _shear_overlap(a_low, a_high, b_low, b_high, period, periodic)
    total = 0
    for image in -1:1
        (periodic || iszero(image)) || continue
        shift = image * period
        total += max(0, min(a_high, b_high + shift) - max(a_low, b_low + shift))
    end
    return total
end

@inline function _shear_overlap(
        a_low, a_high, b_low, b_high, c_low, c_high, period, periodic,
    )
    total = 0
    for b_image in -1:1, c_image in -1:1
        (periodic || (iszero(b_image) && iszero(c_image))) || continue
        b_shift = b_image * period
        c_shift = c_image * period
        total += max(
            0,
            min(a_high, b_high + b_shift, c_high + c_shift) -
                max(a_low, b_low + b_shift, c_low + c_shift),
        )
    end
    return total
end

"""Fill the shear nodes of component `axis`: every owned face of that axis and
every wall face normal to it, on the doubled finest lattice."""
function _shear_nodes!(nodes, mesh, lattice, axis)
    empty!(nodes)
    nx, ny = mesh.block_size
    blocks = mesh.blocks
    for (index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        normal, low, high = _face_lattice_geometry(mesh, face, lattice)
        _, minus_scale = _cell_lattice_bounds(mesh, blocks, face.minus, lattice)
        _, plus_scale = _cell_lattice_bounds(mesh, blocks, face.plus, lattice)
        push!(
            nodes, ShearNode(
                index, 2normal, 2low, 2high,
                2normal - minus_scale[axis], 2normal + plus_scale[axis],
                _cell_index(face.minus, nx, ny), _cell_index(face.plus, nx, ny),
            ),
        )
    end
    for face in mesh.boundary_faces
        face.axis == axis || continue
        normal, low, high = _face_lattice_geometry(mesh, face, lattice)
        _, scale = _cell_lattice_bounds(mesh, blocks, face.cell, lattice)
        extent_low, extent_high = face.side < 0 ?
            (2normal, 2normal + scale[axis]) : (2normal - scale[axis], 2normal)
        push!(
            nodes, ShearNode(
                0, 2normal, 2low, 2high, extent_low, extent_high,
                _cell_index(face.cell, nx, ny), 0,
            ),
        )
    end
    return nodes
end

# First position `p` in `order[first:last]` with `key(order[p]) >= value`, for
# `order` sorted by `key`; `last + 1` if there is none.
@inline function _first_shear_position(key, order, first, last, value)
    while first <= last
        middle = (first + last) >>> 1
        if key(@inbounds order[middle]) < value
            first = middle + 1
        else
            last = middle - 1
        end
    end
    return first
end

"""Find the tangential contacts of component `axis`.

The staggered control volumes of the nodes tile the domain, so the part of
the upper edge of node `a` inside the domain is covered exactly by the lower
edges of the nodes `b` that start there; each overlap is a contact `(a, b)`
whose region spans the overlap along `k` and the centres of `a` and `b`
along `t`. A node on a wall normal to `t` has a wall contact up to the wall.
The contact regions of one component partition the domain."""
function _shear_contacts!(partition, mesh, lattice, axis)
    nodes = partition.nodes
    order = partition.order
    contacts = partition.contacts
    tangent = 3 - axis
    k_period = 2lattice[axis]
    t_period = 2lattice[tangent]
    k_periodic = mesh.periodic[axis]
    t_periodic = mesh.periodic[tangent]
    low_key = index -> mod((@inbounds nodes[index]).low, t_period)
    extent_low_key = index -> mod((@inbounds nodes[index]).extent_low, k_period)
    extent_high_key = index -> begin
        node = @inbounds nodes[index]
        mod(node.extent_low, k_period) + node.extent_high - node.extent_low
    end
    resize!(order, length(nodes))
    for index in eachindex(order)
        @inbounds order[index] = index
    end
    sort!(order; by = index -> (low_key(index), extent_low_key(index)))
    empty!(contacts)
    count = length(order)
    for (index, node) in pairs(nodes)
        centre = (node.low + node.high) ÷ 2
        if !t_periodic && iszero(node.low)
            push!(
                contacts, ShearContact(
                    0, index, node.extent_low, node.extent_high, 0, centre,
                ),
            )
        end
        if !t_periodic && node.high == t_period
            push!(
                contacts, ShearContact(
                    index, 0, node.extent_low, node.extent_high, centre, node.high,
                ),
            )
            continue
        end
        edge = mod(node.high, t_period)
        first = _first_shear_position(low_key, order, 1, count, edge)
        last = _first_shear_position(low_key, order, first, count, edge + 1) - 1
        for image in -1:1
            (k_periodic || iszero(image)) || continue
            shift = image * k_period
            position = _first_shear_position(
                extent_high_key, order, first, last,
                node.extent_low - shift + 1,
            )
            while position <= last
                other_index = @inbounds order[position]
                other_low = extent_low_key(other_index) + shift
                other_low < node.extent_high || break
                other = @inbounds nodes[other_index]
                other_high = extent_high_key(other_index) + shift
                other_centre = (other.low + other.high) ÷ 2
                push!(
                    contacts, ShearContact(
                        index, other_index,
                        max(node.extent_low, other_low),
                        min(node.extent_high, other_high),
                        centre, node.high + other_centre - other.low,
                    ),
                )
                position += 1
            end
        end
    end
    return contacts
end

# CSR lists of the contacts on one side of every node: `lower == node`
# (`upper_side = false`) or `upper == node`.
function _shear_sides!(offsets, list, contacts, number_of_nodes, upper_side)
    resize!(offsets, number_of_nodes + 1)
    fill!(offsets, 0)
    for contact in contacts
        node = upper_side ? contact.upper : contact.lower
        iszero(node) || (@inbounds offsets[node] += 1)
    end
    total = 1
    for node in 1:number_of_nodes
        count = @inbounds offsets[node]
        @inbounds offsets[node] = total
        total += count
    end
    offsets[end] = total
    resize!(list, total - 1)
    for (index, contact) in pairs(contacts)
        node = upper_side ? contact.upper : contact.lower
        iszero(node) && continue
        @inbounds list[offsets[node]] = index
        @inbounds offsets[node] += 1
    end
    for node in number_of_nodes:-1:2
        @inbounds offsets[node] = offsets[node - 1]
    end
    offsets[1] = 1
    return offsets
end

@inline function _push_shear_coefficient!(faces, values, face, value)
    iszero(face) && return nothing
    push!(faces, face)
    push!(values, value)
    return nothing
end

# Control-volume width along `k`; a wall node owns half a cell, so it counts as
# its mirrored full width when the finer node of a contact is chosen.
@inline _shear_width(node::ShearNode) =
    (iszero(node.plus) ? 2 : 1) * (node.extent_high - node.extent_low)

"""Tangential derivative of component `axis` on every contact.

A wall contact is the one-sided difference to the zero no-slip velocity (none
for free slip). Two nodes of equal control-volume width, or a contact whose
narrower node has no other contact on that side, are collinear and use the
two-point difference. Otherwise the narrower node `a` is the anchor of all
contacts on that side: its partners `b` with measures `m_b` are centred,
`sum_b m_b (x_b - x_a)_k = 0`, and share the linear-exact derivative
`±(sum_b m_b u_b / sum_b m_b - u_a) / d`, `d = sum_b m_b |x_b - x_a|_t / sum_b m_b`."""
function _shear_stencils!(partition, mesh, lattice, axis, boundary, ::Type{T}) where {T}
    nodes = partition.nodes
    contacts = partition.contacts
    offsets = partition.stencil_offsets
    faces = partition.stencil_faces
    values = partition.stencil_values
    tangent = 3 - axis
    k_period = 2lattice[axis]
    k_periodic = mesh.periodic[axis]
    spacing = T(mesh.extent[tangent] / lattice[tangent]) / 2
    empty!(faces)
    empty!(values)
    resize!(offsets, length(contacts) + 1)
    for (index, contact) in pairs(contacts)
        @inbounds offsets[index] = length(faces) + 1
        if iszero(contact.lower) || iszero(contact.upper)
            boundary isa NoSlip || continue
            lower_wall = iszero(contact.lower)
            node = nodes[lower_wall ? contact.upper : contact.lower]
            distance = (contact.t_high - contact.t_low) * spacing
            _push_shear_coefficient!(
                faces, values, node.face,
                lower_wall ? inv(distance) : -inv(distance),
            )
            continue
        end
        lower = nodes[contact.lower]
        upper = nodes[contact.upper]
        lower_width = _shear_width(lower)
        upper_width = _shear_width(upper)
        anchor_lower = lower_width < upper_width
        anchor_index = anchor_lower ? contact.lower : contact.upper
        anchor = nodes[anchor_index]
        anchor_width = min(lower_width, upper_width)
        side_offsets, side = anchor_lower ?
            (partition.above_offsets, partition.above) :
            (partition.below_offsets, partition.below)
        group = side_offsets[anchor_index]:(side_offsets[anchor_index + 1] - 1)
        if lower_width == upper_width || length(group) == 1
            _shear_wrap(upper.normal - lower.normal, k_period, k_periodic) == 0 ||
                throw(ArgumentError("unpaired skew shear contact"))
            distance = (contact.t_high - contact.t_low) * spacing
            _push_shear_coefficient!(faces, values, lower.face, -inv(distance))
            _push_shear_coefficient!(faces, values, upper.face, inv(distance))
            continue
        end
        measure = 0
        moment = 0
        separation = 0
        for position in group
            member = contacts[side[position]]
            other_index = anchor_lower ? member.upper : member.lower
            iszero(other_index) && throw(ArgumentError("wall in a shear contact group"))
            other = nodes[other_index]
            _shear_width(other) > anchor_width ||
                throw(ArgumentError("ambiguous shear contact anchor"))
            contact_measure = member.k_high - member.k_low
            measure += contact_measure
            moment += contact_measure *
                _shear_wrap(other.normal - anchor.normal, k_period, k_periodic)
            separation += contact_measure * (member.t_high - member.t_low)
        end
        iszero(moment) || throw(ArgumentError("off-centre shear contact group"))
        distance = T(separation) * spacing / T(measure)
        orientation = anchor_lower ? one(T) : -one(T)
        _push_shear_coefficient!(faces, values, anchor.face, -orientation / distance)
        for position in group
            member = contacts[side[position]]
            other = nodes[anchor_lower ? member.upper : member.lower]
            _push_shear_coefficient!(
                faces, values, other.face,
                orientation * T(member.k_high - member.k_low) /
                    (T(measure) * distance),
            )
        end
    end
    offsets[end] = length(faces) + 1
    return partition
end

function _shear_partition!(partition, mesh, lattice, axis, boundary, ::Type{T}) where {T}
    _shear_nodes!(partition.nodes, mesh, lattice, axis)
    _shear_contacts!(partition, mesh, lattice, axis)
    number_of_nodes = length(partition.nodes)
    _shear_sides!(
        partition.below_offsets, partition.below, partition.contacts,
        number_of_nodes, true,
    )
    _shear_sides!(
        partition.above_offsets, partition.above, partition.contacts,
        number_of_nodes, false,
    )
    _shear_stencils!(partition, mesh, lattice, axis, boundary, T)
    return partition
end

@inline function _shear_contact_cells(nodes, contact)
    lower = iszero(contact.lower) ? (0, 0) :
        (nodes[contact.lower].minus, nodes[contact.lower].plus)
    upper = iszero(contact.upper) ? (0, 0) :
        (nodes[contact.upper].minus, nodes[contact.upper].plus)
    return (lower..., upper...)
end

@inline function _new_shear_cell(cells, position)
    cell = cells[position]
    iszero(cell) && return false
    for previous in 1:(position - 1)
        cells[previous] == cell && return false
    end
    return true
end

# Whether the open interval `low:high` contains `line` or a periodic image.
@inline function _shear_crosses(low, high, line, period, periodic)
    for image in -1:1
        (periodic || iszero(image)) || continue
        low < line + image * period < high && return true
    end
    return false
end

# A quarter of `cell` at a solid-side `corner` of its Cartesian wall belongs to
# the wall rows where the row's contact of the wall-tangential component
# crosses the corner's line along the wall normal: the x-velocity contact for a
# wall normal to y, the y-velocity contact for a wall normal to x.
@inline function _shear_wall_quarter(
        entries, cell, corner, first_contact, second_contact, mesh, lattice,
    )
    for (wall_cell, axis) in entries
        wall_cell == cell || continue
        contact = axis == 2 ? first_contact : second_contact
        _shear_crosses(
            contact.t_low, contact.t_high, 2corner[axis], 2lattice[axis],
            mesh.periodic[axis],
        ) && return true
    end
    return false
end

"""
    _shear_rows!(measures, rows, columns, values, interpolation_rows,
                 interpolation_columns, interpolation_values, cache, mesh,
                 boundary, geometry, ::Type{T})

Append the shear rows of the strain map. Each velocity component `k` has its
own partition of the domain into contact regions between tangentially
adjacent samples (`_shear_contacts!`), with a tangential derivative per
contact (`_shear_stencils!`). A shear row is a nonempty intersection `R1 ∩ R2`
of an `x`-velocity region and a `y`-velocity region: its strain is the sum of
the two derivatives and its measure the open area of the intersection. The
area of the `k` rows covering a contact is the contact measure times the
centre distance, so the transpose of the strain map gives every staggered
control volume the shear flux of its own contact length, also at level
transitions, and the energy remains a sum of squares. On a uniform mesh the
two regions coincide with the vertex dual cell.

The measure is split over the cell quarters (cell, corner) it covers; where
the contact of the wall-tangential velocity crosses a Cartesian embedded wall
of the cell at that corner, the wall rows carry that shear and the quarter is
omitted (`_shear_wall_quarter`). The open fraction scales the rest, and the
row viscosity mixes the cells with these open weights.
"""
function _shear_rows!(
        measures, rows, columns, values, interpolation_rows,
        interpolation_columns, interpolation_values, cache, mesh, boundary,
        geometry, ::Type{T},
    ) where {T}
    lattice = _finest_lattice_size(mesh)
    first_partition, second_partition = cache.shear_partitions
    _shear_partition!(first_partition, mesh, lattice, 1, boundary, T)
    _shear_partition!(second_partition, mesh, lattice, 2, boundary, T)
    wall_vertices = isnothing(geometry) ? nothing :
        _embedded_variable_wall_stress_vertices(geometry, lattice)
    references = cache.cell_references
    number_of_cells = length(references)

    second_nodes = second_partition.nodes
    second_contacts = second_partition.contacts
    registry_offsets = resize!(cache.cell_contact_offsets, number_of_cells + 1)
    fill!(registry_offsets, 0)
    for contact in second_contacts
        cells = _shear_contact_cells(second_nodes, contact)
        for position in 1:4
            _new_shear_cell(cells, position) &&
                (registry_offsets[cells[position]] += 1)
        end
    end
    total = 1
    for cell in 1:number_of_cells
        count = registry_offsets[cell]
        registry_offsets[cell] = total
        total += count
    end
    registry_offsets[end] = total
    registry = resize!(cache.cell_contacts, total - 1)
    for (index, contact) in pairs(second_contacts)
        cells = _shear_contact_cells(second_nodes, contact)
        for position in 1:4
            _new_shear_cell(cells, position) || continue
            registry[registry_offsets[cells[position]]] = index
            registry_offsets[cells[position]] += 1
        end
    end
    for cell in number_of_cells:-1:2
        registry_offsets[cell] = registry_offsets[cell - 1]
    end
    registry_offsets[1] = 1

    marks = resize!(cache.contact_marks, length(second_contacts))
    fill!(marks, 0)
    candidates = cache.contact_candidates
    row_cells = cache.row_cells
    row_weights = cache.row_weights
    x_period, y_period = 2lattice[1], 2lattice[2]
    x_periodic, y_periodic = mesh.periodic
    area_scale = T(mesh.extent[1] / lattice[1]) * T(mesh.extent[2] / lattice[2]) / 4
    first_nodes = first_partition.nodes
    first_offsets = first_partition.stencil_offsets
    second_offsets = second_partition.stencil_offsets
    for (first_index, first_contact) in pairs(first_partition.contacts)
        first_stencil = first_offsets[first_index]:(first_offsets[first_index + 1] - 1)
        cells = _shear_contact_cells(first_nodes, first_contact)
        empty!(candidates)
        for position in 1:4
            _new_shear_cell(cells, position) || continue
            cell = cells[position]
            for entry in registry_offsets[cell]:(registry_offsets[cell + 1] - 1)
                second_index = registry[entry]
                marks[second_index] == first_index && continue
                marks[second_index] = first_index
                push!(candidates, second_index)
            end
        end
        sort!(candidates)
        for second_index in candidates
            second_stencil = second_offsets[second_index]:(second_offsets[second_index + 1] - 1)
            isempty(first_stencil) && isempty(second_stencil) && continue
            # First regions are (k, t) = (x, y), second regions (y, x).
            second_contact = second_contacts[second_index]
            _shear_overlap(
                first_contact.k_low, first_contact.k_high,
                second_contact.t_low, second_contact.t_high, x_period, x_periodic,
            ) > 0 || continue
            _shear_overlap(
                first_contact.t_low, first_contact.t_high,
                second_contact.k_low, second_contact.k_high, y_period, y_periodic,
            ) > 0 || continue
            empty!(row_cells)
            empty!(row_weights)
            for position in 1:4
                _new_shear_cell(cells, position) || continue
                cell = cells[position]
                ref = references[cell]
                low, scale = _cell_lattice_bounds(mesh, mesh.blocks, ref, lattice)
                open = isnothing(geometry) ? one(T) :
                    T(_get(geometry.open_fraction, ref))
                for quarter_x in 0:1, quarter_y in 0:1
                    x_low = 2low[1] + quarter_x * scale[1]
                    y_low = 2low[2] + quarter_y * scale[2]
                    width = _shear_overlap(
                        first_contact.k_low, first_contact.k_high,
                        second_contact.t_low, second_contact.t_high,
                        x_low, x_low + scale[1], x_period, x_periodic,
                    )
                    width > 0 || continue
                    height = _shear_overlap(
                        first_contact.t_low, first_contact.t_high,
                        second_contact.k_low, second_contact.k_high,
                        y_low, y_low + scale[2], y_period, y_periodic,
                    )
                    height > 0 || continue
                    if !isnothing(wall_vertices)
                        corner = (
                            low[1] + quarter_x * scale[1],
                            low[2] + quarter_y * scale[2],
                        )
                        key = (
                            x_periodic ? mod(corner[1], lattice[1]) : corner[1],
                            y_periodic ? mod(corner[2], lattice[2]) : corner[2],
                        )
                        _shear_wall_quarter(
                            get(wall_vertices, key, ()), cell, corner,
                            first_contact, second_contact, mesh, lattice,
                        ) && continue
                    end
                    weight = T(width * height) * area_scale * open
                    weight > zero(T) || continue
                    if !isempty(row_cells) && last(row_cells) == cell
                        row_weights[end] += weight
                    else
                        push!(row_cells, cell)
                        push!(row_weights, weight)
                    end
                end
            end
            open_volume = sum(row_weights; init = zero(T))
            open_volume > zero(T) || continue
            push!(measures, open_volume)
            row = length(measures)
            for (cell, weight) in zip(row_cells, row_weights)
                push!(interpolation_rows, row)
                push!(interpolation_columns, cell)
                push!(interpolation_values, weight / open_volume)
            end
            for (partition, stencil) in (
                    (first_partition, first_stencil),
                    (second_partition, second_stencil),
                )
                for entry in stencil
                    push!(rows, row)
                    push!(columns, partition.stencil_faces[entry])
                    push!(values, partition.stencil_values[entry])
                end
            end
        end
    end
    return measures
end
