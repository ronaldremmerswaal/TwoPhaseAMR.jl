"""
    InterfaceRefinementIndicator(interface; level, buffer_cells=1,
                                 t_init=-Inf, t_end=Inf)

Refine blocks intersecting the interface or its cell-halo neighbourhood to a
fixed level. Pure blocks request coarsening. Aligned interfaces between pure
cells are detected through halo fraction jumps as well as mixed cells. The
criterion is active for `t_init <= time <= t_end`.
"""
struct InterfaceRefinementIndicator{F <: InterfaceField, T} <:
    AbstractRefinementIndicator
    field::F
    level::Int
    buffer_cells::Int
    t_init::T
    t_end::T
end
function InterfaceRefinementIndicator(
        field::InterfaceField{T}; level,
        buffer_cells = 1, t_init = -Inf, t_end = Inf
    ) where {T}
    level isa Integer && !(level isa Bool) &&
        0 <= level <= field.mesh.max_level ||
        throw(ArgumentError("level must be an integer within mesh limits"))
    0 <= buffer_cells <= field.mesh.halo || throw(ArgumentError("buffer_cells must fit the mesh halo"))
    start_time, end_time = _refinement_window(T, t_init, t_end)
    return InterfaceRefinementIndicator(
        field, Int(level), buffer_cells, start_time, end_time
    )
end
_indicator_field(indicator::InterfaceRefinementIndicator) = indicator.field
function _fixed_interface_refinement_labels!(
        labels::AbstractVector{Symbol}, field::InterfaceField,
        level::Int, buffer_cells::Int
    )
    mesh = field.mesh
    length(labels) == nblocks(mesh) || throw(DimensionMismatch("one label is required per block"))
    _refresh_conditional_fraction!(field)
    nx, ny = mesh.block_size
    h, r = mesh.halo, buffer_cells
    _parallel_for(nblocks(mesh); grain = 4) do b
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
    return _fixed_interface_refinement_labels!(
        labels, indicator.field, indicator.level,
        indicator.buffer_cells
    )
end

_refinement_labels_at_time!(
    labels, indicator::InterfaceRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)

"""
    CurvatureRefinementIndicator(interface; cells_per_radius=8,
                                  coarsen_ratio=1/3, minimum_level=0,
                                  maximum_level=interface.mesh.max_level,
                                  t_init=-Inf, t_end=Inf)

Request refinement where `max(dx,dy)*abs(curvature) > 1/cells_per_radius`.
Following the thesis, a mixed cell also requests refinement when no valid LHF
exists or when its height columns reveal a thin filament. Use hysteresis for
coarsening; combine with `InterfaceRefinementIndicator` for a protected band.
The criterion is active for `t_init <= time <= t_end` and enforces its level
bounds independently of the curvature estimate.
"""
struct CurvatureRefinementIndicator{F <: InterfaceField, T} <: AbstractRefinementIndicator
    field::F
    cells_per_radius::T
    coarsen_ratio::T
    minimum_level::Int
    maximum_level::Int
    t_init::T
    t_end::T
end
function CurvatureRefinementIndicator(
        field::InterfaceField;
        cells_per_radius = 8.0, coarsen_ratio = 1 / 3,
        minimum_level = 0, maximum_level = field.mesh.max_level,
        t_init = -Inf, t_end = Inf
    )
    isfinite(cells_per_radius) && cells_per_radius > 0 || throw(ArgumentError("cells_per_radius must be finite and positive"))
    isfinite(coarsen_ratio) && 0 < 2coarsen_ratio < 1 || throw(ArgumentError("coarsen_ratio must satisfy 0 < 2coarsen_ratio < 1"))
    lower_level, upper_level = _refinement_levels(
        field.mesh, minimum_level, maximum_level
    )
    a, b = promote(float(cells_per_radius), float(coarsen_ratio))
    start_time, end_time = _refinement_window(typeof(a), t_init, t_end)
    return CurvatureRefinementIndicator(
        field, a, b, lower_level, upper_level, start_time, end_time
    )
end

"""
    VorticityRefinementIndicator(velocity; refine_tolerance,
        coarsen_tolerance=refine_tolerance/3, minimum_level=0,
        maximum_level=velocity.mesh.max_level, t_init=-Inf, t_end=Inf,
        interface=nothing, phase=:liquid)

Block AMR criterion from the thesis. Refine if any cell in a block has
`max(dx,dy)*abs(vorticity) > refine_tolerance`; coarsen when every cell lies
below `coarsen_tolerance`. The strict hysteresis condition
`2coarsen_tolerance < refine_tolerance` prevents immediate re-refinement after
coarsening. The criterion is active for `t_init <= time <= t_end` and enforces
its level bounds independently of the vorticity estimate.

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
end

function VorticityRefinementIndicator(
        velocity::FaceVelocity{T}; refine_tolerance,
        coarsen_tolerance = refine_tolerance / 3,
        minimum_level = 0, maximum_level = velocity.mesh.max_level,
        t_init = -Inf, t_end = Inf, interface = nothing, phase = :liquid,
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
    return _vorticity_refinement_labels!(labels, indicator, interface, geometry, phase, work)
end

function _vorticity_refinement_labels!(labels, indicator, interface,
        geometry::G, phase, work) where {G}
    # Geometry is specialized once for both curl and the block scan. All
    # phase masks/numerical fits are refreshed; only mesh geometry is cached.
    mesh = indicator.field.mesh
    vorticity = _vorticity_values!(work, indicator.field, interface, geometry, phase)
    nx, ny = mesh.block_size
    block_cells = nx * ny
    _parallel_for(nblocks(mesh); grain = 4) do b
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
    _parallel_for(nblocks(mesh); grain = 4) do b
        block = mesh.blocks[b]
        error = 0.0
        for j in 1:ny, i in 1:nx
            c = interior(field, b)[i, j]
            0 < c < 1 || continue
            valid_lhf, filament = _lhf_refinement_status(
                height_values[b], i + 4, j + 4
            )
            kappa = interior(field.curvature, b)[i, j]
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
    return labels
end


_refinement_labels_at_time!(
    labels, indicator::CurvatureRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)
