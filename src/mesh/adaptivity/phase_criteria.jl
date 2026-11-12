"""
    InterfaceRefinementIndicator(interface; level, buffer_cells=1,
                                 t_init=-Inf, t_end=Inf,
                                 xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)

Refine blocks intersecting the interface or its cell-halo neighbourhood to a
fixed level. Pure blocks request coarsening unless their sibling family must
be retained to prevent the prospective parent's band from refining it again.
Aligned interfaces between pure cells are detected through halo fraction jumps
as well as mixed cells. The criterion is active for `t_init <= time <= t_end`.
Only blocks intersecting the closed spatial region are eligible; outside it
the indicator requests `:coarsen`.
"""
struct InterfaceRefinementIndicator{F <: InterfaceField, T} <:
    AbstractRefinementIndicator
    field::F
    level::Int
    buffer_cells::Int
    t_init::T
    t_end::T
    region::NTuple{4, T}
end
function InterfaceRefinementIndicator(
        field::InterfaceField{T}; level,
        buffer_cells = 1, t_init = -Inf, t_end = Inf,
        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
    ) where {T}
    level isa Integer && !(level isa Bool) &&
        0 <= level <= field.mesh.max_level ||
        throw(ArgumentError("level must be an integer within mesh limits"))
    0 <= buffer_cells <= field.mesh.halo || throw(ArgumentError("buffer_cells must fit the mesh halo"))
    start_time, end_time = _refinement_window(T, t_init, t_end)
    return InterfaceRefinementIndicator(
        field, Int(level), buffer_cells, start_time, end_time,
        _refinement_region(T, xmin, xmax, ymin, ymax)
    )
end
_indicator_field(indicator::InterfaceRefinementIndicator) = indicator.field
# A parent's ring is twice as wide in child cells. Conservatively retain
# siblings if any leaf cells overlapping that ring and its interior span both
# phases; otherwise a merge could immediately be followed by refinement.
# Extrema over the overlapping cells also bound restricted parent fractions.
function _protect_fixed_band_coarsening!(labels, field, target, r)
    iszero(target) && return labels
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    indices = Dict(block.key => b for (b, block) in pairs(mesh.blocks))
    lattice = mesh.base_mesh .<< mesh.max_level
    candidates = Int[]
    floor = field.fraction_floor
    for block in mesh.blocks
        key = block.key
        key.level > 0 && key.level <= target || continue
        iseven(key.i) && iseven(key.j) || continue
        parent = _parent_key(key)
        children = ntuple(i -> get(indices, _child_keys(parent)[i], 0), 4)
        all(!iszero, children) || continue
        all(i -> labels[i] === :coarsen, children) || continue

        low, high = one(eltype(field)), zero(eltype(field))
        # All bounds below use the common finest-level integer cell lattice.
        scale = 1 << (mesh.max_level - key.level)
        lower = ((key.i * nx - 2r) * scale, (key.j * ny - 2r) * scale)
        upper = ((key.i * nx + 2nx + 2r) * scale, (key.j * ny + 2ny + 2r) * scale)
        shifts = ntuple(2) do axis
            mesh.periodic[axis] ?
                (fld(-upper[axis], lattice[axis]):cld(lattice[axis] - lower[axis], lattice[axis])) :
                (0:0)
        end
        for sy in shifts[2], sx in shifts[1]
            low < 1 - floor && high > floor && break
            lo = (max(0, lower[1] + sx * lattice[1]), max(0, lower[2] + sy * lattice[2]))
            hi = (min(lattice[1], upper[1] + sx * lattice[1]), min(lattice[2], upper[2] + sy * lattice[2]))
            lo[1] < hi[1] && lo[2] < hi[2] || continue
            # Outward rounding makes the tree search conservative; the exact
            # lattice intersection below discards any extra candidate blocks.
            physical_lo = ntuple(a -> prevfloat(mesh.origin[a] + mesh.extent[a] * lo[a] / lattice[a]), 2)
            physical_hi = ntuple(a -> nextfloat(mesh.origin[a] + mesh.extent[a] * hi[a] / lattice[a]), 2)
            empty!(candidates)
            _collect_leaf_blocks_in_window!(candidates, mesh.root, indices, physical_lo, physical_hi)
            for index in candidates
                other = mesh.blocks[index].key
                cell_scale = 1 << (mesh.max_level - other.level)
                x0, y0 = other.i * nx * cell_scale, other.j * ny * cell_scale
                imin = max(1, fld(lo[1] - x0, cell_scale) + 1)
                imax = min(nx, cld(hi[1] - x0, cell_scale))
                jmin = max(1, fld(lo[2] - y0, cell_scale) + 1)
                jmax = min(ny, cld(hi[2] - y0, cell_scale))
                imin <= imax && jmin <= jmax || continue
                values = @view field.conditional_fraction.blocks[index][
                    (h + imin):(h + imax), (h + jmin):(h + jmax),
                ]
                cell_low, cell_high = extrema(values)
                low, high = min(low, cell_low), max(high, cell_high)
                low < 1 - floor && high > floor && break
            end
        end
        if low < 1 - floor && high > floor
            for child in children
                labels[child] = :keep
            end
        end
    end
    return labels
end

function _fixed_interface_refinement_labels!(
        labels::AbstractVector{Symbol}, field::InterfaceField,
        level::Int, buffer_cells::Int
    )
    mesh = field.mesh
    length(labels) == nblocks(mesh) || throw(DimensionMismatch("one label is required per block"))
    _refresh_conditional_fraction!(field)
    nx, ny = mesh.block_size
    h, r = mesh.halo, buffer_cells
    _parallel_for(nblocks(mesh); work = _block_work()) do b
        block = mesh.blocks[b]
        values = @view field.conditional_fraction.blocks[b][
            (h + 1 - r):(h + nx + r), (h + 1 - r):(h + ny + r),
        ]
        low, high = extrema(values)
        near = low < 1 - field.fraction_floor &&
            high > field.fraction_floor
        labels[b] = near ? (
                block.key.level < level ? :refine :
                block.key.level > level ? :coarsen : :keep
            ) : :coarsen
    end
    return labels
end

function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::InterfaceRefinementIndicator;
        time = 0
    )
    length(labels) == nblocks(indicator.field.mesh) || throw(
        DimensionMismatch("one label is required per block")
    )
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    _fixed_interface_refinement_labels!(
        labels, indicator.field, indicator.level, indicator.buffer_cells
    )
    _apply_refinement_region!(labels, indicator)
    _protect_fixed_band_coarsening!(
        labels, indicator.field, indicator.level, indicator.buffer_cells
    )
    return _apply_refinement_region!(labels, indicator)
end

_refinement_labels_at_time!(
    labels, indicator::InterfaceRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)

"""
    CurvatureRefinementIndicator(interface; cells_per_radius=8,
                                  coarsen_ratio=1/3, minimum_level=0,
                                  maximum_level=interface.mesh.max_level,
                                  t_init=-Inf, t_end=Inf,
                                  xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)

Request refinement where `max(dx,dy)*abs(curvature) > 1/cells_per_radius`.
Following the thesis, a cell mixed in the open fluid volume also requests
refinement when no valid LHF exists or when its height columns reveal a thin
filament. A block that holds a mixed cell in its own cells or in its one-cell
halo borders the interface, and every wall-cut cell of that block which is
completely liquid in its open volume then counts as an interface cell: the
liquid–gas interface meets the solid there, so its undefined curvature is the
unresolved contact line (#148). The test is per block because the refinement
decision is: a cell-level test would read a finer neighbour's restriction
average at a level step, which is mixed even where the fine interface is
resolved, so the coarse block would refine and its own children would coarsen
again. Liquid-filled cut cells in blocks without a mixed cell lie away from
the interface and stay excluded (#118). Use hysteresis for coarsening; combine
with `InterfaceRefinementIndicator` for a protected band.
The criterion is active for `t_init <= time <= t_end` and enforces its level
bounds independently of the curvature estimate.
Only blocks intersecting the closed spatial region are eligible; outside it
the indicator requests `:coarsen` without enforcing its minimum level.
"""
struct CurvatureRefinementIndicator{F <: InterfaceField, T} <: AbstractRefinementIndicator
    field::F
    cells_per_radius::T
    coarsen_ratio::T
    minimum_level::Int
    maximum_level::Int
    t_init::T
    t_end::T
    region::NTuple{4, T}
end
function CurvatureRefinementIndicator(
        field::InterfaceField;
        cells_per_radius = 8.0, coarsen_ratio = 1 / 3,
        minimum_level = 0, maximum_level = field.mesh.max_level,
        t_init = -Inf, t_end = Inf,
        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
    )
    isfinite(cells_per_radius) && cells_per_radius > 0 || throw(ArgumentError("cells_per_radius must be finite and positive"))
    isfinite(coarsen_ratio) && 0 < 2coarsen_ratio < 1 || throw(ArgumentError("coarsen_ratio must satisfy 0 < 2coarsen_ratio < 1"))
    lower_level, upper_level = _refinement_levels(
        field.mesh, minimum_level, maximum_level
    )
    a, b = promote(float(cells_per_radius), float(coarsen_ratio))
    start_time, end_time = _refinement_window(typeof(a), t_init, t_end)
    return CurvatureRefinementIndicator(
        field, a, b, lower_level, upper_level, start_time, end_time,
        _refinement_region(typeof(a), xmin, xmax, ymin, ymax)
    )
end

"""
    VorticityRefinementIndicator(velocity; refine_tolerance,
        coarsen_tolerance=refine_tolerance/3, minimum_level=0,
        maximum_level=velocity.mesh.max_level, t_init=-Inf, t_end=Inf,
        interface=nothing, phase=:liquid,
        xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf)

Block AMR criterion from the thesis. Refine if any cell in a block has
`max(dx,dy)*abs(vorticity) > refine_tolerance`; coarsen when every cell lies
below `coarsen_tolerance`. The strict hysteresis condition
`2coarsen_tolerance < refine_tolerance` prevents immediate re-refinement after
coarsening. The criterion is active for `t_init <= time <= t_end` and enforces
its level bounds independently of the vorticity estimate.
Only blocks intersecting the closed spatial region are eligible; outside it
the indicator requests `:coarsen` without enforcing its minimum level.

For an intrinsic phase velocity, supply its `interface` and `phase`, or
register the velocity tuple and interface together in `AMRWorkspace`.
Absent-phase cells do not constrain coarsening. The phase curl extends only
valid values into its temporary interface stencil; owned absent velocities
remain NaN.
"""
struct VorticityRefinementIndicator{V <: FaceVelocity, T, I} <:
    AbstractRefinementIndicator
    field::V
    refine_tolerance::T
    coarsen_tolerance::T
    minimum_level::Int
    maximum_level::Int
    t_init::T
    t_end::T
    interface::I
    phase::Symbol
    region::NTuple{4, T}
end

function VorticityRefinementIndicator(
        velocity::FaceVelocity{T}; refine_tolerance,
        coarsen_tolerance = refine_tolerance / 3,
        minimum_level = 0, maximum_level = velocity.mesh.max_level,
        t_init = -Inf, t_end = Inf, interface = nothing, phase = :liquid,
        xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
    ) where {T}
    refine = T(refine_tolerance)
    coarsen = T(coarsen_tolerance)
    isfinite(refine) && refine > zero(T) ||
        throw(ArgumentError("refine_tolerance must be finite and positive"))
    isfinite(coarsen) && zero(T) <= 2coarsen < refine ||
        throw(
        ArgumentError(
            "tolerances must satisfy 0 <= 2coarsen_tolerance < refine_tolerance"
        )
    )
    lower_level, upper_level = _refinement_levels(
        velocity.mesh, minimum_level, maximum_level
    )
    start_time, end_time = _refinement_window(T, t_init, t_end)
    phase in (:liquid, :gas) || throw(ArgumentError("phase must be :liquid or :gas"))
    isnothing(interface) || interface.mesh === velocity.mesh ||
        throw(ArgumentError("vorticity velocity and interface must share a mesh"))
    return VorticityRefinementIndicator(
        velocity, refine, coarsen, lower_level, upper_level,
        start_time, end_time, interface, phase,
        _refinement_region(T, xmin, xmax, ymin, ymax),
    )
end

_indicator_field(indicator::VorticityRefinementIndicator) = indicator.field

function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::VorticityRefinementIndicator;
        time = 0, interface = indicator.interface, phase = indicator.phase,
        _vorticity_workspace = nothing,
    )
    mesh = indicator.field.mesh
    length(labels) == nblocks(mesh) ||
        throw(DimensionMismatch("one label is required per block"))
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    work = isnothing(_vorticity_workspace) ?
        VorticityWorkspace(eltype(indicator.field)) : _vorticity_workspace
    geometry = isnothing(interface) ? nothing : interface.geometry
    _vorticity_refinement_labels!(labels, indicator, interface, geometry, phase, work)
    return _apply_refinement_region!(labels, indicator)
end

function _vorticity_refinement_labels!(
        labels, indicator, interface,
        geometry::G, phase, work
    ) where {G}
    # Geometry is specialized once for both curl and the block scan. All
    # phase masks/numerical fits are refreshed; only mesh geometry is cached.
    mesh = indicator.field.mesh
    vorticity = _vorticity_values!(work, indicator.field, interface, geometry, phase)
    nx, ny = mesh.block_size
    block_cells = nx * ny
    _parallel_for(nblocks(mesh); work = _block_work()) do b
        block = mesh.blocks[b]
        offset = (b - 1) * block_cells
        error = zero(eltype(vorticity))
        unresolved = false
        h = maximum(block.spacing)
        for local_index in 1:block_cells
            i = mod(local_index - 1, nx) + 1
            j = (local_index - 1) ÷ nx + 1
            if !isnothing(interface)
                _phase_cell_is_active(interface, geometry, CellRef(b, i, j), phase) || continue
            end
            value = @inbounds vorticity[offset + local_index]
            if isnan(value)
                # No valid two-dimensional phase stencil: do not use an
                # invented vorticity, nor coarsen an unresolved active cell.
                unresolved = true
                continue
            end
            error = max(
                error,
                h * abs(value)
            )
        end
        level = block.key.level
        labels[b] = if level < indicator.minimum_level
            :refine
        elseif level > indicator.maximum_level
            :coarsen
        elseif error > indicator.refine_tolerance &&
                level < indicator.maximum_level
            :refine
        elseif !unresolved && error < indicator.coarsen_tolerance &&
                _indicator_can_coarsen(level, indicator.minimum_level)
            :coarsen
        else
            :keep
        end
    end
    return labels
end

_refinement_labels_at_time!(
    labels, indicator::VorticityRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)
_indicator_field(indicator::CurvatureRefinementIndicator) = indicator.field

"""
    _wetted_cut_cell(conditional, open, i, j, halo)

Return whether interior cell `(i, j)` of one block is a wall-cut cell that is
completely liquid inside its open volume. `conditional` and `open` are the
block's haloed arrays.

At a contact line the liquid–gas interface meets the solid inside a cell that
the wall has cut down to its wetted part: that cell holds liquid in all of its
open volume (`conditional == 1`), so its curvature is undefined (#148). On its
own that does not make it an interface cell, because every wetted cell of the
solid boundary looks the same (#118); the caller therefore counts these cells
only in a block that also holds a mixed cell.
"""
@inline function _wetted_cut_cell(conditional, open, i, j, halo)
    ii, jj = i + halo, j + halo
    return @inbounds open[ii, jj] < 1 && conditional[ii, jj] == 1
end

# A family may only coarsen when its parent would not immediately be refined
# again. The interface band protects its coarsening families the same way. A
# contact-line cell is an interface cell in the coarse view even where the fine
# interface is resolved, because the interface crossing a coarse cell makes
# that block border the interface; merging the family would then refine it
# straight back and cycle the mesh (#148). The window is the family plus a
# `2r`-cell ring, so a contact line just outside the family still protects it,
# while a wetted solid boundary with no interface in reach coarsens (#118).
# Covered cells are excluded from the phase extrema: their conditional
# fraction is zero by construction, not gas, so counting them would make every
# window that crosses the solid span both phases.
function _protect_contact_line_coarsening!(labels, field, wetted, r)
    mesh = field.mesh
    geometry = field.geometry
    isnothing(geometry) && return labels
    nx, ny = mesh.block_size
    h = mesh.halo
    floor = field.fraction_floor
    indices = Dict(block.key => b for (b, block) in pairs(mesh.blocks))
    lattice = mesh.base_mesh .<< mesh.max_level
    candidates = Int[]
    for block in mesh.blocks
        key = block.key
        key.level > 0 || continue
        iseven(key.i) && iseven(key.j) || continue
        parent = _parent_key(key)
        children = ntuple(i -> get(indices, _child_keys(parent)[i], 0), 4)
        all(!iszero, children) || continue
        all(i -> labels[i] === :coarsen, children) || continue
        any(i -> wetted[i], children) || continue

        # `key` is the lower-left child, so its two-block span is the parent's
        # and both bounds live on the common finest-level cell lattice.
        scale = 1 << (mesh.max_level - key.level)
        lower = ((key.i * nx - 2r) * scale, (key.j * ny - 2r) * scale)
        upper = (
            (key.i * nx + 2nx + 2r) * scale,
            (key.j * ny + 2ny + 2r) * scale,
        )
        shifts = ntuple(2) do axis
            mesh.periodic[axis] ?
                (
                    fld(-upper[axis], lattice[axis]):
                    cld(lattice[axis] - lower[axis], lattice[axis])
                ) : (0:0)
        end
        low, high = one(eltype(field)), zero(eltype(field))
        for sy in shifts[2], sx in shifts[1]
            low < 1 - floor && high > floor && break
            lo = (
                max(0, lower[1] + sx * lattice[1]),
                max(0, lower[2] + sy * lattice[2]),
            )
            hi = (
                min(lattice[1], upper[1] + sx * lattice[1]),
                min(lattice[2], upper[2] + sy * lattice[2]),
            )
            lo[1] < hi[1] && lo[2] < hi[2] || continue
            physical_lo = ntuple(
                a -> prevfloat(mesh.origin[a] + mesh.extent[a] * lo[a] / lattice[a]), 2
            )
            physical_hi = ntuple(
                a -> nextfloat(mesh.origin[a] + mesh.extent[a] * hi[a] / lattice[a]), 2
            )
            empty!(candidates)
            _collect_leaf_blocks_in_window!(
                candidates, mesh.root, indices, physical_lo, physical_hi
            )
            for index in candidates
                other = mesh.blocks[index].key
                cell_scale = 1 << (mesh.max_level - other.level)
                x0, y0 = other.i * nx * cell_scale, other.j * ny * cell_scale
                imin = max(1, fld(lo[1] - x0, cell_scale) + 1)
                imax = min(nx, cld(hi[1] - x0, cell_scale))
                jmin = max(1, fld(lo[2] - y0, cell_scale) + 1)
                jmax = min(ny, cld(hi[2] - y0, cell_scale))
                imin <= imax && jmin <= jmax || continue
                conditional = @inbounds field.conditional_fraction.blocks[index]
                open = @inbounds geometry.open_fraction.blocks[index]
                @inbounds for j in (h + jmin):(h + jmax), i in (h + imin):(h + imax)
                    # A covered cell holds no fluid: its conditional fraction
                    # is zero by construction, not gas (#118).
                    open[i, j] > 0 || continue
                    value = conditional[i, j]
                    low = min(low, value)
                    high = max(high, value)
                end
                low < 1 - floor && high > floor && break
            end
        end
        low < 1 - floor && high > floor || continue
        for child in children
            labels[child] = :keep
        end
    end
    return labels
end

function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::CurvatureRefinementIndicator;
        time = 0
    )
    field = indicator.field;mesh = field.mesh
    length(labels) == nblocks(mesh) || throw(DimensionMismatch("one label is required per block"))
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    reconstruct!(field)
    height_values = _fill_height_values!(field)
    nx, ny = mesh.block_size
    h = mesh.halo
    geometry = field.geometry
    wetted = fill(false, nblocks(mesh))
    _parallel_for(nblocks(mesh); work = _block_work()) do b
        block = mesh.blocks[b]
        conditional = @inbounds field.conditional_fraction.blocks[b]
        open = isnothing(geometry) ? nothing :
            @inbounds geometry.open_fraction.blocks[b]
        # Refinement is a block decision, so the contact-line test is one too:
        # a block holding a mixed cell in its own cells or its one-cell halo
        # borders the interface, and its wetted cut cells are where that
        # interface meets the solid (#148). Testing cell adjacency instead
        # would read a finer neighbour's restriction average at a level step:
        # that average is mixed even where the fine interface is resolved, so
        # the coarse block would refine while its own children, seeing the
        # resolved values, coarsen again, and the mesh would cycle.
        near_interface = any(c -> 0 < c < 1, conditional)
        error = 0.0
        for j in 1:ny, i in 1:nx
            c = @inbounds conditional[i + h, j + h]
            wetted_cell = !isnothing(open) &&
                _wetted_cut_cell(conditional, open, i, j, h)
            wetted_cell && (wetted[b] = true)
            interface_cell = 0 < c < 1 || (wetted_cell && near_interface)
            interface_cell || continue
            valid_lhf, filament = _lhf_refinement_status(
                height_values[b], i + 4, j + 4
            )
            kappa = @inbounds field.curvature.blocks[b][i + h, j + h]
            unresolved = !valid_lhf || filament || !isfinite(kappa)
            error = max(
                error,
                unresolved ? Inf : maximum(block.spacing) *
                    abs(kappa) * indicator.cells_per_radius
            )
        end
        level = block.key.level
        labels[b] = if level < indicator.minimum_level
            :refine
        elseif level > indicator.maximum_level
            :coarsen
        elseif error > 1 && level < indicator.maximum_level
            :refine
        elseif error < indicator.coarsen_ratio &&
                _indicator_can_coarsen(level, indicator.minimum_level)
            :coarsen
        else
            :keep
        end
    end
    _protect_contact_line_coarsening!(labels, field, wetted, h)
    return _apply_refinement_region!(labels, indicator)
end


_refinement_labels_at_time!(
    labels, indicator::CurvatureRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)
