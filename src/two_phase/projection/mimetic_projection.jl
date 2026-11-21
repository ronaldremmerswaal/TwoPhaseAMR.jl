# Kinetic-energy-orthogonal two-velocity projection in the mimetic cut-cell
# metric (`MimeticProjection`). Meshes of square cells, uniform or adaptive
# (2:1 graded), with or without embedded solids. See
# docs/techref/chapters/10_projection.tex, "Mimetic cut-cell projection".

import SparseArrays: CHOLMOD
import SparseArrays.LibSuiteSparse: CHOLMOD_LONG, CHOLMOD_REAL, CHOLMOD_DOUBLE

"""
    MimeticProjectionSolver(interface; density=(1, 1))

Configuration of the mimetic cut-cell projection on `interface` (with the
embedded solid of `interface.geometry`, if any). Throws `ArgumentError` for
non-square cells and element types other than `Float64`.
"""
struct MimeticProjectionSolver{T, I}
    interface::I
    density::NTuple{2, T}
end

function MimeticProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
    ) where {T}
    T === Float64 || throw(ArgumentError("MimeticProjection requires Float64"))
    rho = (T(density[1]), T(density[2]))
    all(value -> isfinite(value) && value > zero(T), rho) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    _mimetic_square_cells(interface.mesh)
    return MimeticProjectionSolver{T, typeof(interface)}(interface, rho)
end

"""Sparse `LDLᵀ` factorisation of a symmetric positive definite or
quasi-definite matrix. CHOLMOD computes it (simplicial, without BLAS) and
keeps its symbolic analysis while the sparsity pattern of the factorised
matrix is unchanged (the result is bitwise that of a fresh factorisation).
The factor (`L` with `D` on its diagonal, the diagonal first in each column,
in `pointers`, `rows`, `entries`) and the fill-reducing permutation are copied
into reused vectors (`_mimetic_ldlt_copy!`), and `_mimetic_ldlt_solve!` solves
with them in serial loops without allocating. `work` is the scratch of the
solve: one factorisation is not solved with from two threads at once."""
mutable struct _MimeticLDLT{T}
    factor::Union{Nothing, CHOLMOD.Factor{T, Int}}
    colptr::Vector{Int}
    rowval::Vector{Int}
    pointers::Vector{Int}
    rows::Vector{Int}
    entries::Vector{T}
    permutation::Vector{Int}
    work::Vector{T}
end

_MimeticLDLT(::Type{T}) where {T} = _MimeticLDLT{T}(nothing, Int[], Int[], Int[], Int[], T[], Int[], T[])

"""Factorise `matrix` (a `Symmetric` view of a sparse matrix) into `ldlt`."""
function _mimetic_ldlt!(ldlt::_MimeticLDLT{T}, matrix::Symmetric) where {T}
    data = parent(matrix)
    if isnothing(ldlt.factor) || ldlt.colptr != data.colptr || ldlt.rowval != rowvals(data)
        ldlt.factor = CHOLMOD.ldlt(matrix)
        copy!(ldlt.colptr, data.colptr)
        copy!(ldlt.rowval, rowvals(data))
    else
        CHOLMOD.ldlt!(ldlt.factor, matrix)
    end
    _mimetic_ldlt_copy!(ldlt, size(data, 1)) || _mimetic_ldlt_public_copy!(ldlt)
    resize!(ldlt.work, size(data, 1))
    return ldlt
end

"""Copy the factor of `ldlt.factor` (dimension `n`) out of CHOLMOD's own
storage, the `cholmod_factor` struct, without allocating. The struct is not
public API, so the copy checks its kind first: a real double-precision
simplicial `LDLᵀ` factor with `Int` indices of dimension `n` that factorised
completely (`minor == n`), each column holding its diagonal first. Returns
`false`, the copy unusable, when a check fails; `_mimetic_ldlt!` then copies
through the public API (`_mimetic_ldlt_public_copy!`)."""
function _mimetic_ldlt_copy!(ldlt::_MimeticLDLT{T}, n) where {T}
    T === Float64 || return false
    factor = ldlt.factor
    GC.@preserve factor begin
        stored = unsafe_load(factor.ptr)
        kind = stored.itype == CHOLMOD_LONG && stored.xtype == CHOLMOD_REAL &&
            stored.dtype == CHOLMOD_DOUBLE && iszero(stored.is_ll) && iszero(stored.is_super)
        kind && stored.n == n && stored.minor == n || return false
        # Column `j` holds `nz[j]` entries from `p[j]` (zero-based; the columns
        # are not necessarily in order in memory).
        start, row, value = Ptr{Int}(stored.p), Ptr{Int}(stored.i), Ptr{T}(stored.x)
        count, order = Ptr{Int}(stored.nz), Ptr{Int}(stored.Perm)
        pointers = resize!(ldlt.pointers, n + 1)
        pointers[1] = 1
        for column in 1:n
            pointers[column + 1] = pointers[column] + unsafe_load(count, column)
        end
        rows = resize!(ldlt.rows, pointers[n + 1] - 1)
        entries = resize!(ldlt.entries, pointers[n + 1] - 1)
        for column in 1:n
            offset = unsafe_load(start, column) - pointers[column] + 1
            for entry in pointers[column]:(pointers[column + 1] - 1)
                rows[entry] = unsafe_load(row, entry + offset) + 1
                entries[entry] = unsafe_load(value, entry + offset)
            end
            pointers[column] < pointers[column + 1] && rows[pointers[column]] == column || return false
        end
        permutation = resize!(ldlt.permutation, n)
        for index in 1:n
            permutation[index] = unsafe_load(order, index) + 1
        end
    end
    return true
end

"""Copy the factor of `ldlt.factor` through CHOLMOD's public API (`LD` as a
sparse matrix, rows sorted, so the diagonal comes first, and `p`); allocates."""
function _mimetic_ldlt_public_copy!(ldlt::_MimeticLDLT)
    lower = sparse(ldlt.factor.LD)
    copy!(ldlt.pointers, lower.colptr)
    copy!(ldlt.rows, rowvals(lower))
    copy!(ldlt.entries, nonzeros(lower))
    copy!(ldlt.permutation, ldlt.factor.p)
    return ldlt
end

"""`x = A⁻¹ b` with the factorisation `ldlt` of `A = Pᵀ L D Lᵀ P`."""
function _mimetic_ldlt_solve!(x, ldlt::_MimeticLDLT, b)
    pointers, rows, entries = ldlt.pointers, ldlt.rows, ldlt.entries
    permutation = ldlt.permutation
    y = ldlt.work
    for index in eachindex(permutation)
        y[index] = b[permutation[index]]
    end
    for column in eachindex(y)
        value = y[column]
        for pointer in (pointers[column] + 1):(pointers[column + 1] - 1)
            y[rows[pointer]] -= entries[pointer] * value
        end
    end
    for column in eachindex(y)
        y[column] /= entries[pointers[column]]
    end
    for column in reverse(eachindex(y))
        value = y[column]
        for pointer in (pointers[column] + 1):(pointers[column + 1] - 1)
            value -= entries[pointer] * y[rows[pointer]]
        end
        y[column] = value
    end
    for index in eachindex(permutation)
        x[permutation[index]] = y[index]
    end
    return x
end

"""Storage of the mimetic viscous step (`_mimetic_viscous_step!`): conjugate
gradient vectors on the phase values, face fields of the mixture, the band
block of the operator with its factorisation, and the number of conjugate
gradient restarts with the lifted preconditioner alone (`restarts`)."""
mutable struct _MimeticViscousWork{T}
    values::Vector{T}
    solution::Vector{T}
    rhs::Vector{T}
    residual::Vector{T}
    direction::Vector{T}
    preconditioned::Vector{T}
    product::Vector{T}
    face_mass::Vector{T}
    face_density::Vector{T}
    face_sum::Vector{T}
    face_stress::Vector{T}
    face_correction::Vector{T}
    face_diagonal::Vector{T}
    band_product::Vector{T}
    band_residual::Vector{T}
    drag::Vector{T}
    stress_work::Vector{T}
    band_rows::Vector{Int}
    band_columns::Vector{Int}
    band_values::Vector{T}
    band_rhs::Vector{T}
    band_solution::Vector{T}
    band_ldlt::_MimeticLDLT{T}
    restarts::Int
end

_MimeticViscousWork(::Type{T}) where {T} = _MimeticViscousWork{T}(
    T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[],
    Int[], Int[], T[], T[], T[], _MimeticLDLT(T), 0,
)

"""Reusable fields and the most recently assembled mimetic system.

`pressure_indices` are the cells whose pressure is unknown and
`pressure_anchors` the anchor cell of each (one per open-face component of
the active cells, its cell of largest open volume); `anchor` maps every
active cell to its anchor (zero for a cell without open volume) and
`cell_volume` is the open volume of a cell (`h²` without a solid);
`reduced_index` maps a cell to its position in `pressure_indices` (zero for
an anchor or a cell without open volume). `band_ldlt` factorises the band
block of `𝓜`, `saddle_ldlt` the band-cell saddle-point matrix of the
pressure preconditioner on the band cells `band_cells` (positions in
`pressure_indices`; `cell_band_position` maps a cell to its position among
them, zero off them; see `_mimetic_band_cell_saddle!`)."""
mutable struct MimeticProjectionWorkspace{T, M <: BlockTreeMesh}
    liquid_fraction_x::FaceField{1, T, M}
    liquid_fraction_y::FaceField{2, T, M}
    face_data::Union{Nothing, _ProjectionFaceData{T}}
    topology::Union{Nothing, _MimeticTopology{T}}
    references::Vector{CellRef}
    assembly::_MimeticAssembly{T}
    band_ldlt::_MimeticLDLT{T}
    saddle_ldlt::_MimeticLDLT{T}
    band_cells::Vector{Int}
    reduced_index::Vector{Int}
    cell_band_position::Vector{Int}
    saddle_colptr::Vector{Int}
    saddle_rowval::Vector{Int}
    saddle_nzval::Vector{T}
    saddle_rhs::Vector{T}
    saddle_solution::Vector{T}
    divergence_band::SparseMatrixCSC{T, Int}
    divergence_band_transpose::SparseMatrixCSC{T, Int}
    diagonal_laplacian::SparseMatrixCSC{T, Int}
    laplacian_topology::Union{Nothing, _MimeticTopology{T}}
    laplacian_positions::Vector{Int}
    transpose_cursors::Vector{Int}
    boundary_flux::Vector{T}
    boundary_aperture::Vector{T}
    boundary_low::Vector{Bool}
    gridline::Vector{Int8}
    solid::_MimeticSolidData{T}
    gauge_topology::Union{Nothing, _MimeticTopology{T}}
    gauge_revision::UInt
    pressure_indices::Vector{Int}
    pressure_anchors::Vector{Int}
    anchor::Vector{Int}
    cell_volume::Vector{T}
    anchor_pressure::Vector{T}
    component_stack::Vector{Int}
    incidence::SparseMatrixCSC{T, Int}
    incidence_counts::Vector{Int}
    preconditioner_coefficient::Vector{T}
    pressure_operator::Any
    multigrid::Any
    preconditioner::Any
    preconditioner_topology::Union{Nothing, _MimeticTopology{T}}
    predictor::Vector{T}
    velocity::Vector{T}
    metric_predictor::Vector{T}
    metric_velocity::Vector{T}
    multiplier::Vector{T}
    cell_residual::Vector{T}
    cell_scratch::Vector{T}
    cell_product::Vector{T}
    band_scratch::Vector{T}
    band_solution::Vector{T}
    reduced_rhs::Vector{T}
    reduced_solution::Vector{T}
    cg_residual::Vector{T}
    cg_direction::Vector{T}
    cg_preconditioned::Vector{T}
    cg_product::Vector{T}
    cg_schur::Vector{T}
    cg_scratch::Vector{T}
    cg_correction::Vector{T}
    force_rhs::Vector{T}
    force_solution::Vector{T}
    cell_curvature::Vector{T}
    curvature_weight::Vector{T}
    cell_potential::Vector{T}
    viscous::_MimeticViscousWork{T}
end

"""
    MimeticProjectionWorkspace(solver::MimeticProjectionSolver)

Allocate reusable storage for [`project!`](@ref) with a mimetic projection.
"""
function MimeticProjectionWorkspace(solver::MimeticProjectionSolver{T}) where {T}
    mesh = solver.interface.mesh
    return MimeticProjectionWorkspace{T, typeof(mesh)}(
        FaceField{1}(mesh, T), FaceField{2}(mesh, T), nothing, nothing,
        CellRef[], _MimeticAssembly(T), _MimeticLDLT(T), _MimeticLDLT(T), Int[], Int[], Int[],
        Int[], Int[], T[], T[], T[], spzeros(T, 0, 0), spzeros(T, 0, 0),
        spzeros(T, 0, 0), nothing, Int[], Int[],
        T[], T[], Bool[], Int8[], _MimeticSolidData(T), nothing, UInt(0),
        Int[], Int[], Int[], T[], T[], Int[], spzeros(T, 0, 0), Int[],
        T[], nothing, nothing, nothing, nothing,
        T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[],
        T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[],
        _MimeticViscousWork(T),
    )
end

"""Diagnostics of one mimetic cut-cell projection.

`mixture_volume_defect` is `dt * max(abs(div(u_mix)))` of the projected state
(with an embedded solid the cell outflow divided by `max(V, h²/100)`, `V` the
open volume); `target_defect` the same relative to an inherited mixture-flux target.
`energy_identity_defect` is `(E(u*) - E(u⁺) - E(u* - u⁺)) / E(u*)` in the norm
of the metric `𝓜`; on grid-line interface faces `E(u*)` and `E(u* - u⁺)`
include the energy of the phase-velocity jump that identifying the two phase
values removes. `momentum_change` is the change of the first-moment
momentum `(𝓜e_i + Ce_i)ᵀu` (`C` the wall coupling of the metric, zero away
from walls) and `momentum_wall_force` the term `dt (D e_i)ᵀ p` of the interior
faces; without values coupled to a wall the two are equal, with them they
differ by `(Ce_i)ᵀ(u⁺ - u*)`, the momentum the walls take up; `transported_momentum_change` is the change of the transported
momentum `e_iᵀWu` with the diagonal phase masses `W`, which this projection
does not conserve, and `transported_momentum` its value before projection.
`wall_cells` counts the cut cells with a side on the domain boundary and
`solid_cells` the cells cut by an embedded solid (cut by the interface or
not), whose solid edges enter the metric.
`defective_cells` counts the cut cells whose Gram tensor has no consistent
block (least-squares fallback, see [`MimeticProjection`](@ref)); `lumped_values`
the phase values that no block and no tie gives inertia (lumped mass).
"""
struct MimeticProjectionStats{T}
    converged::Bool
    iterations::Int
    initial_residual::T
    final_residual::T
    mixture_divergence::T
    mixture_volume_defect::T
    target_defect::T
    energy_identity_defect::T
    momentum_change::NTuple{2, T}
    momentum_wall_force::NTuple{2, T}
    transported_momentum_change::NTuple{2, T}
    transported_momentum::NTuple{2, T}
    pressure_unknowns::Int
    band_unknowns::Int
    phase_values::Int
    cut_cells::Int
    wall_cells::Int
    solid_cells::Int
    slaved_values::Int
    stabilised_values::Int
    lumped_values::Int
    defective_cells::Int
    gridline_faces::Int
    preconditioner::Symbol
end

_dimensionless_residual(stats::MimeticProjectionStats) = stats.mixture_volume_defect
_cut_cell_counts(stats::MimeticProjectionStats) = (stats.defective_cells, stats.lumped_values)

function Base.show(io::IO, stats::MimeticProjectionStats)
    return print(
        io, "MimeticProjectionStats(converged=", stats.converged,
        ", iterations=", stats.iterations,
        ", mixture_volume_defect=", stats.mixture_volume_defect,
        ", energy_identity_defect=", stats.energy_identity_defect,
        ", momentum_change=", stats.momentum_change,
        ", transported_momentum_change=", stats.transported_momentum_change,
        ", unknowns=(`p`=", stats.pressure_unknowns,
        ", band=", stats.band_unknowns, ")",
        ", cut_cells=", stats.cut_cells,
        ", wall_cells=", stats.wall_cells,
        ", solid_cells=", stats.solid_cells,
        ", slaved_values=", stats.slaved_values,
        ", stabilised_values=", stats.stabilised_values,
        ", lumped_values=", stats.lumped_values,
        ", defective_cells=", stats.defective_cells,
        ", gridline_faces=", stats.gridline_faces, ")",
    )
end

"""Assemble the metric and the reduced operators for the current interface.

The assembly is reused while the interface fractions (hence apertures), the
embedded solid and the mesh topology are unchanged.
"""
function _mimetic_assemble!(
        solver::MimeticProjectionSolver{T}, workspace::MimeticProjectionWorkspace{T},
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    geometry = interface.geometry
    topology_changed = !_mimetic_topology_current(workspace.topology, mesh)
    if topology_changed
        workspace.topology = _mimetic_topology(mesh)
        workspace.references = _cell_references(mesh)
        _resize_face_field!(workspace.liquid_fraction_x, mesh)
        _resize_face_field!(workspace.liquid_fraction_y, mesh)
    end
    solid_changed = !isnothing(geometry) && (
        topology_changed || workspace.gauge_topology !== workspace.topology ||
            workspace.gauge_revision != _interface_geometry_revision(geometry)
    )
    data = _projection_face_data(solver, workspace; storage = workspace.face_data)
    workspace.face_data = data
    assembly = workspace.assembly
    revision = interface.phase_aperture_updates[]
    !topology_changed && !solid_changed && assembly.revision == revision && return assembly
    flags = _mimetic_face_sides!(assembly.flags, interface, workspace.topology, data.liquid_aperture)
    gridline = _mimetic_gridline_faces!(workspace.gridline, interface)
    boundary_aperture = resize!(workspace.boundary_aperture, length(mesh.boundary_faces))
    boundary_low = resize!(workspace.boundary_low, length(mesh.boundary_faces))
    for (index, face) in pairs(mesh.boundary_faces)
        boundary_aperture[index] = _phase_boundary_face_apertures(interface, geometry, index)[1]
        fraction = _conditional_fraction(interface, face.cell)
        boundary_low[index] = zero(T) < fraction < one(T) && first(
            _mimetic_trace_vote(
                interface, face.cell, face.axis, face.side > 0 ? 0.5 : -0.5, (-0.5, 0.5),
            ),
        ) > 0
    end
    solid = isnothing(geometry) ? nothing : _mimetic_solid_data!(
            workspace.solid, interface, geometry, workspace.references, solid_changed,
        )
    _mimetic_assemble_from_data!(
        workspace, mesh, data, flags, boundary_aperture,
        boundary_low, solver.density; gridline, solid, geometry,
    )
    assembly.revision = revision
    return assembly
end

"""Fill the embedded-solid data of the metric (`_MimeticSolidData`): with
`static`, the data of the fixed solid on the current mesh (open apertures,
solid ends, open dual volumes, open fractions and centroids of the open
polygons), and on every call the liquid PLIC lines of the cells."""
function _mimetic_solid_data!(
        solid::_MimeticSolidData{T}, interface, geometry, references, static,
    ) where {T}
    mesh = interface.mesh
    cells = length(references)
    if static
        faces, boundaries = length(mesh.faces), length(mesh.boundary_faces)
        copyto!(resize!(solid.open, faces), geometry.aperture)
        copyto!(resize!(solid.boundary_open, boundaries), geometry.boundary_aperture)
        copyto!(resize!(solid.face_volume, faces), geometry.face_dual_volumes)
        resize!(solid.low, faces)
        for (index, face) in pairs(mesh.faces)
            solid.low[index] = _mimetic_solid_low(geometry.solid, (face.minus, face.plus), face.axis)
        end
        resize!(solid.boundary_low, boundaries)
        for (index, face) in pairs(mesh.boundary_faces)
            solid.boundary_low[index] = _mimetic_solid_low(geometry.solid, (face.cell,), face.axis)
        end
        for values in (solid.cell_open, solid.centroid_x, solid.centroid_y)
            resize!(values, cells)
        end
        for cell in 1:cells
            reference = references[cell]
            solid.cell_open[cell] = _get(geometry.open_fraction, reference)
            area, first_x, first_y = _polygon_area_moments(_open_cell_polygon(geometry, reference))
            solid.centroid_x[cell] = area > zero(T) ? first_x / area : zero(T)
            solid.centroid_y[cell] = area > zero(T) ? first_y / area : zero(T)
        end
    end
    for values in (solid.line_x, solid.line_y, solid.line_d, solid.fraction)
        resize!(values, cells)
    end
    for cell in 1:cells
        reference = references[cell]
        solid.line_x[cell] = _get(interface.normal_x, reference)
        solid.line_y[cell] = _get(interface.normal_y, reference)
        solid.line_d[cell] = _get(interface.intercept, reference)
        solid.fraction[cell] = _conditional_fraction(interface, reference)
    end
    return solid
end

"""Whether the solid closes the low tangential end of a face of axis `axis`
between the cells `references`: the sign of the sum of the tangential
components of the unit normals of their solid PLIC planes (the solid is
`n ⋅ x ≤ d`, at the low end for a positive component); the low end without a
plane."""
function _mimetic_solid_low(solid, references, axis)
    T = eltype(solid)
    vote = zero(T)
    for reference in references
        fraction = _get(solid.fraction, reference)
        zero(T) < fraction < one(T) || continue
        nx, ny = _get(solid.normal_x, reference), _get(solid.normal_y, reference)
        vote += (axis == 1 ? ny : nx) / hypot(nx, ny)
    end
    return vote >= zero(T)
end

"""Assemble `𝓜`, factorize its band block, build the reduced divergence
operators and factorize the band-cell saddle-point matrix of the pressure
preconditioner (`_mimetic_band_cell_saddle!`) from given face geometry (see
`_assemble_cut_cell_metric!`; `solid` the data of an embedded solid,
`geometry` the solid for the pressure gauge)."""
function _mimetic_assemble_from_data!(
        workspace::MimeticProjectionWorkspace{T}, mesh, data, flags,
        boundary_aperture, boundary_low, density; gridline = nothing,
        solid = nothing, geometry = nothing,
    ) where {T}
    if !_mimetic_topology_current(workspace.topology, mesh)
        workspace.topology = _mimetic_topology(mesh)
        workspace.references = _cell_references(mesh)
    end
    topology = workspace.topology
    _mimetic_gauge!(workspace, mesh, geometry)
    assembly = _assemble_cut_cell_metric!(
        workspace.assembly, mesh, topology, data, flags,
        boundary_aperture, boundary_low, density; gridline, solid,
    )
    isempty(assembly.band) || _mimetic_ldlt!(workspace.band_ldlt, Symmetric(assembly.band_matrix, :U))
    cells = size(topology.sides, 2)
    _mimetic_cell_operators!(workspace, assembly)
    resize!(workspace.boundary_flux, cells)
    fill!(workspace.boundary_flux, zero(T))
    count = length(assembly.band)
    resize!(workspace.band_scratch, count)
    resize!(workspace.band_solution, count)
    resize!(workspace.cell_residual, cells)
    resize!(workspace.cell_scratch, cells)
    resize!(workspace.cell_product, cells)
    resize!(workspace.multiplier, cells)
    # Face coefficient of the preconditioning pressure operator `D W⁻¹ Dᵀ`, in
    # units of 1/rho: Σ_π (a|f|)² / W_r with W_r the diagonal phase mass, per
    # the conductance |f|/d of the face (d the distance of the cell centres
    # normal to it, |f| d its dual volume).
    beta = resize!(workspace.preconditioner_coefficient, length(mesh.faces))
    fill!(beta, zero(T))
    for dof in eachindex(assembly.dof_face)
        aperture = assembly.dof_aperture[dof]
        iszero(aperture) && continue
        face = assembly.dof_face[dof]
        volume = isnothing(solid) ? mesh.face_dual_volumes[face] : solid.face_volume[face]
        beta[face] += aperture^2 * volume / assembly.dof_mass[dof]
    end
    _mimetic_band_cell_saddle!(workspace)
    return assembly
end

"""Assemble the divergence `D_b` of the band values (cells × band values:
`+|f|a` on the minus cell, `-|f|a` on the plus cell), its transpose, and the
cell matrix `L = D_d W_d⁻¹ D_dᵀ` of the diagonal values directly in their
storage. `L` keeps the pattern of the cell–face incidence of the topology
(explicit zeros on the faces without a diagonal value); the pattern and the
positions of the four entries of every face are rebuilt only when the
topology changes. The entries are summed in the order of the values, as a
triplet assembly would."""
function _mimetic_cell_operators!(workspace::MimeticProjectionWorkspace{T}, assembly) where {T}
    topology = workspace.topology
    cells = size(topology.sides, 2)
    faces = size(topology.face_cells, 2)
    if workspace.laplacian_topology !== topology
        pattern_rows = Vector{Int}(undef, 4faces)
        pattern_columns = Vector{Int}(undef, 4faces)
        for face in 1:faces
            minus, plus = topology.face_cells[1, face], topology.face_cells[2, face]
            pattern_rows[(4face - 3):4face] .= (minus, plus, minus, plus)
            pattern_columns[(4face - 3):4face] .= (minus, plus, plus, minus)
        end
        pattern = sparse(pattern_rows, pattern_columns, zeros(T, 4faces), cells, cells)
        positions = resize!(workspace.laplacian_positions, 4faces)
        for entry in eachindex(pattern_rows)
            range = nzrange(pattern, pattern_columns[entry])
            offset = searchsortedfirst(view(rowvals(pattern), range), pattern_rows[entry])
            positions[entry] = range[offset]
        end
        workspace.diagonal_laplacian = pattern
        workspace.laplacian_topology = topology
    end
    laplacian = nonzeros(workspace.diagonal_laplacian)
    fill!(laplacian, zero(T))
    positions = workspace.laplacian_positions
    count = length(assembly.band)
    divergence = workspace.divergence_band
    pointers = resize!(divergence.colptr, count + 1)
    rows = resize!(rowvals(divergence), 2count)
    entries = resize!(nonzeros(divergence), 2count)
    pointers[1] = 1
    cursor = 0
    for dof in eachindex(assembly.dof_face)
        position = assembly.band_position[dof]
        aperture = assembly.dof_aperture[dof]
        if !iszero(aperture)
            face = assembly.dof_face[dof]
            weight = topology.area[face] * aperture
            minus = topology.face_cells[1, face]
            plus = topology.face_cells[2, face]
            if position == 0
                stiffness = weight^2 / assembly.diagonal[dof]
                laplacian[positions[4face - 3]] += stiffness
                laplacian[positions[4face - 2]] += stiffness
                laplacian[positions[4face - 1]] -= stiffness
                laplacian[positions[4face]] -= stiffness
            elseif minus != plus
                # A face joining a cell to itself carries no divergence.
                low, high = minmax(minus, plus)
                rows[cursor + 1], rows[cursor + 2] = low, high
                entries[cursor + 1] = low == minus ? weight : -weight
                entries[cursor + 2] = high == minus ? weight : -weight
                cursor += 2
            end
        end
        position > 0 && (pointers[position + 1] = cursor + 1)
    end
    resize!(rows, cursor)
    resize!(entries, cursor)
    workspace.divergence_band = SparseMatrixCSC(cells, count, pointers, rows, entries)
    workspace.divergence_band_transpose = _cmdgfm_transpose!(
        workspace.divergence_band_transpose, workspace.divergence_band,
        workspace.transpose_cursors,
    )
    return workspace
end

"""Relative regularisation `ε` of the cell block of the band-cell saddle-point
matrix (`_mimetic_band_cell_saddle!`)."""
const _MIMETIC_SADDLE_REGULARIZATION = 1.0e-4

"""Band cells of the pressure preconditioner and the `LDLᵀ` factorisation of
their saddle-point matrix

    K = [𝓜_bb  -D_bcᵀ; -D_bc  -(L_cc + εΔ)].

The band cells are the cells with a pressure unknown adjacent to a band value
with an aperture: the rows of the Schur complement `S = L + D_b 𝓜_bb⁻¹ D_bᵀ`
(`L` the pressure operator of the diagonal values) that differ from the
preconditioning operator. `D_bc` is the divergence of the band values on
them, `L_cc` the principal block of `L`, `Δ` the diagonal of
`L + D_b diag(𝓜_bb)⁻¹ D_bᵀ` on them and `ε = _MIMETIC_SADDLE_REGULARIZATION`.
Eliminating the band values gives `-(S_cc + εΔ)`, `S_cc` the band-cell block
of `S`. `L_cc` is singular wherever a band cell has band values on all its
sides, which is common (64 of the 154 band cells of a drop at `n = 32` and
densities `(1000, 1)`); with `ε > 0` the matrix is symmetric quasi-definite,
so its `LDLᵀ` factorisation exists for every symmetric ordering. `ε` serves
only that factorisation: the preconditioner is symmetric positive definite
for `ε = 0` as well (`_mimetic_pressure_precondition!`). The upper triangle
of `K` is assembled column by column in reused storage (no triplets; the
entries and their order are those of a triplet assembly that skips the zero
entries of `L`), the cells in increasing order, so the matrix does not depend
on the thread count. Without band cells nothing is factorised."""
function _mimetic_band_cell_saddle!(workspace::MimeticProjectionWorkspace{T}) where {T}
    assembly = workspace.assembly
    band = assembly.band
    cells = length(workspace.anchor)
    reduced = fill!(resize!(workspace.reduced_index, cells), 0)
    for (index, cell) in pairs(workspace.pressure_indices)
        reduced[cell] = index
    end
    band_cells = empty!(workspace.band_cells)
    isempty(band) && return workspace
    divergence = workspace.divergence_band
    transpose = workspace.divergence_band_transpose
    laplacian = workspace.diagonal_laplacian
    metric = assembly.band_matrix
    position = fill!(resize!(workspace.cell_band_position, cells), 0)
    for cell in rowvals(divergence)
        reduced[cell] > 0 && (position[cell] = 1)
    end
    for cell in 1:cells
        position[cell] > 0 || continue
        push!(band_cells, reduced[cell])
        position[cell] = length(band_cells)
    end
    isempty(band_cells) && return workspace
    count = length(band)
    dimension = count + length(band_cells)
    # Δ, accumulated in the solution buffer before the factorisation: the
    # band part here, the diagonal of `L` with the cell columns below.
    delta = fill!(resize!(workspace.saddle_solution, length(band_cells)), zero(T))
    for column in 1:count
        diagonal = zero(T)
        for pointer in nzrange(metric, column)
            rowvals(metric)[pointer] == column && (diagonal = nonzeros(metric)[pointer])
        end
        for pointer in nzrange(divergence, column)
            k = position[rowvals(divergence)[pointer]]
            k > 0 && (delta[k] += nonzeros(divergence)[pointer]^2 / diagonal)
        end
    end
    pointers = resize!(workspace.saddle_colptr, dimension + 1)
    rows = empty!(workspace.saddle_rowval)
    values = empty!(workspace.saddle_nzval)
    pointers[1] = 1
    # Band columns: the upper triangle of 𝓜_bb.
    for column in 1:count
        for pointer in nzrange(metric, column)
            row = rowvals(metric)[pointer]
            row <= column || continue
            push!(rows, row)
            push!(values, nonzeros(metric)[pointer])
        end
        pointers[column + 1] = length(rows) + 1
    end
    # Cell columns: -D_bcᵀ (the band values of the cell, rows in increasing
    # order in the transpose), then -L_cc above and on the diagonal (the rows
    # of a column of `L` in increasing cell order, so in increasing `k`) with
    # -εΔ on the diagonal.
    epsilon = T(_MIMETIC_SADDLE_REGULARIZATION)
    for (k, index) in pairs(band_cells)
        cell = workspace.pressure_indices[index]
        for pointer in nzrange(transpose, cell)
            push!(rows, rowvals(transpose)[pointer])
            push!(values, -nonzeros(transpose)[pointer])
        end
        laplacian_diagonal = zero(T)
        for pointer in nzrange(laplacian, cell)
            other = position[rowvals(laplacian)[pointer]]
            value = nonzeros(laplacian)[pointer]
            if other == k
                laplacian_diagonal = value
            elseif 0 < other < k && !iszero(value)
                push!(rows, count + other)
                push!(values, -value)
            end
        end
        iszero(laplacian_diagonal) || (delta[k] += laplacian_diagonal)
        push!(rows, count + k)
        push!(
            values, iszero(laplacian_diagonal) ? -epsilon * delta[k] :
                -laplacian_diagonal + -epsilon * delta[k],
        )
        pointers[count + k + 1] = length(rows) + 1
    end
    _mimetic_ldlt!(
        workspace.saddle_ldlt,
        Symmetric(SparseMatrixCSC(dimension, dimension, pointers, rows, values), :U),
    )
    resize!(workspace.saddle_rhs, dimension)
    resize!(workspace.saddle_solution, dimension)
    return workspace
end

"""Pressure gauge of the mimetic solve: the unknown cells, their anchors, the
anchor of every cell and the cell volumes (see
`MimeticProjectionWorkspace`). Without a solid every cell but the first is
unknown; with one, the cells with open volume form components connected by
open faces (`_projection_gauge_partition!`), each anchored at its cell of
largest open volume (the first of equal ones), so that no sliver cell holds
the gauge. Rebuilt when the topology or the solid changes."""
function _mimetic_gauge!(workspace::MimeticProjectionWorkspace{T}, mesh, geometry) where {T}
    topology = workspace.topology
    revision = isnothing(geometry) ? UInt(0) : _interface_geometry_revision(geometry)
    workspace.gauge_topology === topology && workspace.gauge_revision == revision && return workspace
    cells = size(topology.sides, 2)
    indices, anchors, anchor = workspace.pressure_indices, workspace.pressure_anchors, workspace.anchor
    volume = resize!(workspace.cell_volume, cells)
    if isnothing(geometry)
        resize!(indices, cells - 1) .= 2:cells
        fill!(resize!(anchors, cells - 1), 1)
        fill!(resize!(anchor, cells), 1)
        for cell in 1:cells
            volume[cell] = topology.spacing[cell]^2
        end
    else
        copyto!(volume, _geometry_cell_volumes(geometry))
        workspace.incidence = _projection_incidence!(
            workspace.incidence, workspace.incidence_counts, mesh,
        )
        # `anchor` serves as the label scratch of the partition.
        _projection_gauge_partition!(
            indices, anchors, anchor, workspace.component_stack, workspace.incidence,
            mesh, geometry.active_cells, geometry.aperture,
        )
        # `anchor[cell]`: the root (smallest cell) of its component.
        fill!(resize!(anchor, cells), 0)
        for cell in 1:cells
            geometry.active_cells[cell] && (anchor[cell] = cell)
        end
        for (index, cell) in pairs(indices)
            anchor[cell] = anchors[index]
        end
        # `largest[root]`: the cell of largest open volume of that component.
        largest = fill!(resize!(workspace.component_stack, cells), 0)
        for cell in 1:cells
            root = anchor[cell]
            root > 0 || continue
            best = largest[root]
            (best == 0 || volume[cell] > volume[best]) && (largest[root] = cell)
        end
        empty!(indices)
        empty!(anchors)
        for cell in 1:cells
            root = anchor[cell]
            root > 0 || continue
            anchor[cell] = largest[root]
            if anchor[cell] != cell
                push!(indices, cell)
                push!(anchors, anchor[cell])
            end
        end
    end
    workspace.gauge_topology = topology
    workspace.gauge_revision = revision
    return workspace
end

function _mimetic_boundary_flux!(flux, solver, workspace, velocity)
    interface = solver.interface
    topology = workspace.topology
    fill!(flux, zero(eltype(flux)))
    nx, ny = interface.mesh.block_size
    for (index, face) in pairs(interface.mesh.boundary_faces)
        liquid, gas = _phase_boundary_face_apertures(interface, interface.geometry, index)
        cell = _cell_index(face.cell, nx, ny)
        flux[cell] += face.side * face.area * _projection_phase_sum(
            liquid, velocity[1].boundary[index], gas, velocity[2].boundary[index],
        )
    end
    return flux
end

# y = D 𝓜⁻¹ Dᵀ x on the cells; `x` has zero at the anchor.
function _mimetic_schur_full!(y, workspace, x)
    mul!(y, workspace.diagonal_laplacian, x)
    if !isempty(workspace.assembly.band)
        band = workspace.band_scratch
        mul!(band, workspace.divergence_band_transpose, x)
        _mimetic_ldlt_solve!(workspace.band_solution, workspace.band_ldlt, band)
        mul!(y, workspace.divergence_band, workspace.band_solution, true, true)
    end
    return y
end

"""`y = S x` with the gauge-reduced Schur complement `S = D 𝓜⁻¹ Dᵀ` of the
mimetic saddle-point system on the pressure unknowns `pressure_indices` (zero
pressure at the anchors), applied matrix-free: the band block of `𝓜` with its
factorisation, the diagonal part through an assembled cell matrix. Its cell
buffers (`cell_scratch`, `cell_product`) are its own, so the outflow
(`cell_residual`) of `_mimetic_solve!` survives a solve."""
function _mimetic_schur!(y, workspace, x)
    T = eltype(y)
    full = workspace.cell_scratch
    fill!(full, zero(T))
    indices = workspace.pressure_indices
    for (index, cell) in pairs(indices)
        full[cell] = x[index]
    end
    result = workspace.cell_product
    _mimetic_schur_full!(result, workspace, full)
    for (index, cell) in pairs(indices)
        y[index] = result[cell]
    end
    return y
end

"""Add `y = (S_cc + εΔ)⁻¹ g_c` to the band cells of `z` and, unless `w` is
`nothing`, `S E y` to `w` (`g` on the pressure unknowns, `g_c` its band-cell
part, `E` the extension by zero; see `_mimetic_band_cell_saddle!`). The band
values of the saddle-point solve are `v = 𝓜_bb⁻¹ D_bcᵀ y`, so `S E y = L E y +
D_b v` costs two sparse products on the band."""
function _mimetic_band_cell_solve!(z, w, workspace, g)
    count = length(workspace.assembly.band)
    rhs = workspace.saddle_rhs
    solution = workspace.saddle_solution
    band_cells = workspace.band_cells
    for index in 1:count
        rhs[index] = zero(eltype(rhs))
    end
    for (k, index) in pairs(band_cells)
        rhs[count + k] = -g[index]
    end
    _mimetic_ldlt_solve!(solution, workspace.saddle_ldlt, rhs)
    for (k, index) in pairs(band_cells)
        z[index] += solution[count + k]
    end
    isnothing(w) && return z
    reduced = workspace.reduced_index
    laplacian = workspace.diagonal_laplacian
    for (k, index) in pairs(band_cells)
        y = solution[count + k]
        for pointer in nzrange(laplacian, workspace.pressure_indices[index])
            target = reduced[rowvals(laplacian)[pointer]]
            target > 0 && (w[target] = muladd(nonzeros(laplacian)[pointer], y, w[target]))
        end
    end
    divergence = workspace.divergence_band
    for column in 1:count
        v = solution[column]
        for pointer in nzrange(divergence, column)
            target = reduced[rowvals(divergence)[pointer]]
            target > 0 && (w[target] = muladd(nonzeros(divergence)[pointer], v, w[target]))
        end
    end
    return z
end

"""Pressure preconditioner `z = B r`: one V-cycle `M` of the preconditioning
operator (`_mimetic_preconditioner!`) inside a symmetric multiplicative
correction with the band cells, `C = E (S_cc + εΔ)⁻¹ Eᵀ`
(`_mimetic_band_cell_saddle!`),

    z₁ = C r,   z₂ = z₁ + M (r - S z₁),   z = z₂ + C (r - S z₂).

`S` differs from the operator of `M` only in the rows of the band cells, on
which `C` is exact up to `ε`. From `I - BS = (I - CS)(I - MS)(I - CS)`,
`B = 2C - CSC + (I - CS) M (I - SC)`. With `Y = (S_cc + εΔ)⁻¹`,
`2C - CSC = E Y (S_cc + 2εΔ) Y Eᵀ` is positive semi-definite, as `S_cc`, a
principal block of the symmetric positive definite `S`, is positive definite,
and the second term is positive on the kernel of `C`, where `I - SC` is the
identity: `B` is symmetric positive definite for every symmetric positive
definite `M`. The argument holds for `ε = 0` too; `ε > 0` is needed only for
the `LDLᵀ` factorisation of the saddle-point matrix (`L_cc` is often singular).
`S z₁` comes from the saddle-point solve (`_mimetic_band_cell_solve!`), so one
application costs one V-cycle, one Schur product and two band-cell solves.
Without band cells `z = M r`."""
function _mimetic_pressure_precondition!(z, workspace, preconditioner, r)
    T = eltype(z)
    isempty(workspace.band_cells) && return ldiv!(z, preconditioner, r)
    w = workspace.cg_schur
    fill!(z, zero(T))
    fill!(w, zero(T))
    _mimetic_band_cell_solve!(z, w, workspace, r)
    scratch = workspace.cg_scratch
    correction = workspace.cg_correction
    scratch .= r .- w
    ldiv!(correction, preconditioner, scratch)
    z .+= correction
    _mimetic_schur!(scratch, workspace, correction)
    scratch .= r .- w .- scratch
    _mimetic_band_cell_solve!(z, nothing, workspace, scratch)
    return z
end

"""Preconditioned conjugate gradients for `S x = b` (`b = workspace.reduced_rhs`,
`x = workspace.reduced_solution` from zero) with the preconditioner
`_mimetic_pressure_precondition!`, stopped when the preconditioned residual
norm `√(rᵀBr)` falls to `rtol` times its initial value or vanishes, or after
`itmax` iterations. A negative or non-finite `rᵀBr`, or a non-positive or
non-finite `pᵀSp`, throws (`S` or `B` not symmetric positive definite, as
Krylov.jl's `cg` threw). `Sp` is a Schur product (`_mimetic_schur!`), not the
recurrence of the directions from products `Sz` of the preconditioner: their
band-cell part comes from the quasi-definite saddle-point solve, exact to only
round-off times its conditioning (up to `6e-12` relative measured), and the
recursive residual would drift from `b - Sx` by as much (a film at rest at
densities `(1000, 1)` then moved its gas at `3e-15` and was set in motion
within two steps). Reductions are serial, so the bits do not depend on the
Julia thread count. Returns the iterations."""
function _mimetic_pressure_cg!(workspace, preconditioner, rtol, itmax)
    x = workspace.reduced_solution
    r = workspace.cg_residual
    p = workspace.cg_direction
    z = workspace.cg_preconditioned
    q = workspace.cg_product
    T = eltype(x)
    fill!(x, zero(T))
    copyto!(r, workspace.reduced_rhs)
    _mimetic_pressure_precondition!(z, workspace, preconditioner, r)
    gamma = _mimetic_dot(r, z)
    _mimetic_cg_check(isfinite(gamma) && gamma >= zero(T), "rᵀBr", gamma)
    iszero(gamma) && return 0
    target = rtol * sqrt(gamma)
    copyto!(p, z)
    iterations = 0
    while iterations < itmax
        _mimetic_schur!(q, workspace, p)
        curvature = _mimetic_dot(p, q)
        _mimetic_cg_check(isfinite(curvature) && curvature > zero(T), "pᵀSp", curvature)
        alpha = gamma / curvature
        x .+= alpha .* p
        r .-= alpha .* q
        iterations += 1
        _mimetic_pressure_precondition!(z, workspace, preconditioner, r)
        next = _mimetic_dot(r, z)
        _mimetic_cg_check(isfinite(next) && next >= zero(T), "rᵀBr", next)
        sqrt(next) > target || break
        p .= z .+ (next / gamma) .* p
        gamma = next
    end
    return iterations
end

@inline _mimetic_cg_check(valid, name, value) = valid || _mimetic_cg_breakdown(name, value)

@noinline _mimetic_cg_breakdown(name, value) = error(
    "MimeticProjection pressure solve: $name = $value in conjugate gradients; " *
        "the Schur complement or its preconditioner is not symmetric positive definite",
)

"""Phase values of `velocity` on the degrees of freedom of the assembly.

Returns the kinetic energy `Σ m_l m_g / (m_l + m_g) (u^l - u^g)²` that
identifying the phase values of the grid-line faces removes."""
function _mimetic_gather!(values, assembly, velocity)
    resize!(values, length(assembly.dof_face))
    removed = zero(eltype(values))
    for dof in eachindex(assembly.dof_face)
        face = assembly.dof_face[dof]
        phase = assembly.dof_phase[dof]
        value = if phase == 3
            # An identified grid-line value starts from the 𝓜-orthogonal
            # (mass-weighted) projection of the two phase values onto u^l = u^g.
            liquid, gas = velocity[1].interior[face], velocity[2].interior[face]
            weight = assembly.dof_liquid_weight[dof]
            removed += assembly.dof_mass[dof] * weight * (one(weight) - weight) *
                (liquid - gas)^2
            weight * liquid + (one(weight) - weight) * gas
        else
            velocity[phase].interior[face]
        end
        isfinite(value) || throw(
            ArgumentError("non-finite active phase velocity on face $face"),
        )
        values[dof] = value
    end
    return removed
end

"""Write (or with `add`, add) the value of `dof` to its phase velocity; an
identified grid-line value goes to both phases."""
@inline function _mimetic_scatter!(velocity, assembly, dof, value, add)
    face = assembly.dof_face[dof]
    phase = assembly.dof_phase[dof]
    for target in (phase == 3 ? (1, 2) : (phase,))
        values = velocity[target].interior
        if add
            isnan(values[face]) || (values[face] += value)
        else
            values[face] = value
        end
    end
    return nothing
end

"""Mark the interior faces between a pure liquid and a pure gas cell (`1`:
liquid minus cell, `-1`: liquid plus cell); with an embedded solid only open
faces between cells with open volume. Cell fractions are snapped to exactly
zero and one below the interface's fraction floor, so the test is exact."""
function _mimetic_gridline_faces!(gridline, interface)
    mesh = interface.mesh
    geometry = interface.geometry
    resize!(gridline, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        minus = _conditional_fraction(interface, face.minus)
        plus = _conditional_fraction(interface, face.plus)
        gridline[index] = minus == 1 && iszero(plus) ? Int8(1) :
            iszero(minus) && plus == 1 ? Int8(-1) : Int8(0)
        isnothing(geometry) && continue
        open = geometry.aperture[index] > 0 &&
            _get(geometry.open_fraction, face.minus) > 0 &&
            _get(geometry.open_fraction, face.plus) > 0
        open || (gridline[index] = Int8(0))
    end
    return gridline
end

"""`result = 𝓜 values` with the assembled metric."""
function _mimetic_metric_product!(result, workspace, values)
    assembly = workspace.assembly
    resize!(result, length(values))
    for dof in eachindex(values)
        result[dof] = assembly.diagonal[dof] * values[dof]
    end
    band = assembly.band
    isempty(band) && return result
    scratch = workspace.band_scratch
    solution = workspace.band_solution
    for (position, dof) in pairs(band)
        scratch[position] = values[dof]
    end
    mul!(solution, assembly.band_matrix, scratch)
    for (position, dof) in pairs(band)
        result[dof] = solution[position]
    end
    return result
end

"""`result = 𝓜⁻¹ rhs` with the assembled metric (the force map)."""
function _mimetic_metric_solve!(result, workspace, rhs)
    assembly = workspace.assembly
    resize!(result, length(rhs))
    for dof in eachindex(rhs)
        diagonal = assembly.diagonal[dof]
        result[dof] = iszero(diagonal) ? zero(eltype(result)) : rhs[dof] / diagonal
    end
    band = assembly.band
    isempty(band) && return result
    scratch = workspace.band_scratch
    solution = workspace.band_solution
    for (position, dof) in pairs(band)
        scratch[position] = rhs[dof]
    end
    _mimetic_ldlt_solve!(solution, workspace.band_ldlt, scratch)
    for (position, dof) in pairs(band)
        result[dof] = solution[position]
    end
    return result
end

"""Velocities `z⁺ = z* + 𝓜⁻¹ Dᵀ λ` and the cell outflow of the result."""
function _mimetic_recover!(workspace, multiplier)
    assembly = workspace.assembly
    topology = workspace.topology
    velocity = workspace.velocity
    copyto!(velocity, workspace.predictor)
    band = assembly.band
    if !isempty(band)
        mul!(workspace.band_scratch, workspace.divergence_band_transpose, multiplier)
        _mimetic_ldlt_solve!(workspace.band_solution, workspace.band_ldlt, workspace.band_scratch)
        for (position, dof) in pairs(band)
            velocity[dof] += workspace.band_solution[position]
        end
    end
    for dof in eachindex(velocity)
        assembly.band_position[dof] > 0 && continue
        aperture = assembly.dof_aperture[dof]
        iszero(aperture) && continue
        face = assembly.dof_face[dof]
        weight = topology.area[face] * aperture
        minus = topology.face_cells[1, face]
        plus = topology.face_cells[2, face]
        velocity[dof] += weight * (multiplier[minus] - multiplier[plus]) /
            assembly.diagonal[dof]
    end
    return velocity
end

"""Cell outflow `D z + b` of the phase values `values`."""
function _mimetic_outflow!(outflow, workspace, values)
    assembly = workspace.assembly
    topology = workspace.topology
    copyto!(outflow, workspace.boundary_flux)
    for dof in eachindex(values)
        aperture = assembly.dof_aperture[dof]
        iszero(aperture) && continue
        face = assembly.dof_face[dof]
        flux = topology.area[face] * aperture * values[dof]
        outflow[topology.face_cells[1, face]] += flux
        outflow[topology.face_cells[2, face]] -= flux
    end
    return outflow
end

"""Gauge-reduced V-cycle of the pressure operator with the face coefficient
`workspace.preconditioner_coefficient` (by default `D W⁻¹ Dᵀ`, the pressure
operator of the diagonal phase-mass metric; with an embedded solid the
cut-cell pressure operator of its open cells). The operator and its multigrid
hierarchy are rebuilt when the mesh topology changes (a regrid)."""
function _mimetic_preconditioner!(solver, workspace)
    interface = solver.interface
    geometry = interface.geometry
    operator = workspace.pressure_operator
    fresh = !(operator isa EllipticOperator) || operator.mesh !== interface.mesh ||
        workspace.preconditioner_topology !== workspace.topology
    if fresh
        operator = isnothing(geometry) ? PoissonOperator(interface; density = solver.density) :
            PoissonOperator(interface, geometry; density = solver.density)
        workspace.pressure_operator = operator
        workspace.preconditioner_topology = workspace.topology
    end
    _replace_face_coefficient(operator, workspace.preconditioner_coefficient)
    multigrid = workspace.multigrid
    if !fresh && multigrid isa TreeGeometricMultigridPreconditioner
        _refresh_geometric_multigrid!(multigrid, operator)
    else
        multigrid = TreeGeometricMultigridPreconditioner(operator; geometry)
        workspace.multigrid = multigrid
    end
    previous = workspace.preconditioner
    workspace.preconditioner = _cmdgfm_reduced_pressure(
        multigrid, operator, workspace.pressure_indices, workspace.pressure_anchors, geometry;
        reuse = !fresh && previous isa CMDGFMReducedPressurePreconditioner ? previous : nothing,
    )
    return workspace.preconditioner
end

"""
    project!(velocity, pressure, solver::MimeticProjectionSolver, dt, workspace;
             tolerance, max_iterations)

Project the liquid and gas face velocities onto vanishing aperture-weighted
mixture divergence, orthogonally in the mimetic cut-cell kinetic metric `𝓜`
(`MimeticProjection`). The projection is absolute: `velocity` is the full
predictor, and `pressure` is overwritten with the projection pressure, whose
value in the anchor cell of each open component (the first cell without a
solid, the cell of largest open volume with one) is kept; cells without open volume keep theirs. Forces must therefore not subtract a
pressure gradient from the predictor; any `𝓜⁻¹Dᵀ` gradient in the predictor
is removed exactly.

The saddle-point system for the band face values and the pressure is solved
with the conjugate-gradient method on its pressure Schur complement, which is
applied matrix-free with a sparse `LDLᵀ` factorization of the band block of
`𝓜`. The preconditioner is one geometric-multigrid V-cycle of the pressure
operator `D W⁻¹ Dᵀ` of the diagonal phase-mass metric inside a symmetric
multiplicative correction that solves the Schur complement on the cells next
to the band values (`_mimetic_pressure_precondition!`). All reductions are
serial and the `LDLᵀ` factorizations simplicial, so the result does not
depend on the Julia thread count. It does depend on the BLAS thread count,
through the UMFPACK `LU` factorization of the coarsest multigrid level (as in
the CMDGFM projection). `_inherited_cell_flux` and `_reference_dt` have the
meaning of the CMDGFM projection.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::MimeticProjectionSolver{T}, dt,
        workspace::MimeticProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500,
        _inherited_cell_flux = nothing, _reference_dt = dt,
    ) where {T}
    interface = solver.interface
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    pressure.mesh === interface.mesh || throw(
        ArgumentError("mimetic projection arguments must share a mesh"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    tolerance > zero(tolerance) || throw(ArgumentError("tolerance must be strictly positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be strictly positive"))
    isfinite(_reference_dt) && _reference_dt >= dt || throw(
        ArgumentError("_reference_dt must be finite and at least dt"),
    )
    _mark_inactive_phase_velocities!(velocity, interface)
    assembly = _mimetic_assemble!(solver, workspace)
    _mimetic_boundary_flux!(workspace.boundary_flux, solver, workspace, velocity)
    removed_energy = _mimetic_gather!(workspace.predictor, assembly, velocity)
    preconditioner = _mimetic_preconditioner!(solver, workspace)
    # Warm start from the previous pressure, relative to the anchor of each
    # component.
    cells = length(workspace.references)
    multiplier = workspace.multiplier
    anchor = workspace.anchor
    base = resize!(workspace.anchor_pressure, cells)
    for cell in 1:cells
        if iszero(anchor[cell])
            base[cell] = multiplier[cell] = zero(T)
            continue
        end
        base[cell] = _get(pressure, workspace.references[anchor[cell]])
        multiplier[cell] = T(dt) * (_get(pressure, workspace.references[cell]) - base[cell])
    end
    weight = isnothing(_inherited_cell_flux) ? zero(T) : one(T) - T(dt) / T(_reference_dt)
    result = _mimetic_solve!(
        workspace, T(dt), preconditioner, _inherited_cell_flux, weight;
        tolerance = T(tolerance), max_iterations, removed_energy,
    )
    projected = workspace.velocity
    for dof in eachindex(projected)
        _mimetic_scatter!(velocity, assembly, dof, projected[dof], false)
    end
    _mark_inactive_phase_velocities!(velocity, interface)
    for cell in 1:cells
        iszero(anchor[cell]) && continue
        _set!(pressure, base[cell] + multiplier[cell] / T(dt), workspace.references[cell])
    end
    fill_halo!(pressure)
    return MimeticProjectionStats(
        result.target_defect <= T(tolerance), result.iterations,
        result.initial_residual, result.final_residual, result.mixture_divergence,
        T(dt) * result.mixture_divergence, result.target_defect,
        result.energy_identity_defect, result.momentum_change,
        result.momentum_wall_force, result.transported_momentum_change,
        result.transported_momentum, length(workspace.pressure_indices), length(assembly.band),
        length(projected), assembly.cut_cells, assembly.wall_cells, assembly.solid_cells,
        assembly.slaved_values,
        assembly.stabilised_values, assembly.lumped_values, assembly.defective_cells,
        assembly.gridline_faces,
        :band_corrected_multigrid,
    )
end

"""Smallest volume, in units of `h²`, by which the volume defect of a cell is
divided: the defect of a sliver cell is its volume defect per hundredth of
the cell, so the round-off of its fluxes does not hold the solve above its
tolerance. It does not act without a solid."""
const _MIMETIC_VOLUME_FLOOR = 1.0e-2

"""Solve the projection for the gathered predictor `workspace.predictor`.

`workspace.multiplier` holds the initial guess of `λ = dt p` (zero in the
anchor cells and the cells without open volume) and returns the solution; `workspace.velocity` returns the
projected phase values. The target outflow is `-weight` times the inherited
(negative integrated) cell flux. `removed_energy` is the jump energy removed
by gathering (see `_mimetic_gather!`); it enters the energy identity. The
volume defects divide by `max(V, _MIMETIC_VOLUME_FLOOR h²)` with the open
volume `V`.
"""
function _mimetic_solve!(
        workspace::MimeticProjectionWorkspace{T}, dt, preconditioner, inherited,
        weight; tolerance, max_iterations, removed_energy = zero(T),
    ) where {T}
    assembly = workspace.assembly
    topology = workspace.topology
    cells = size(topology.sides, 2)
    predictor = workspace.predictor
    resize!(workspace.velocity, length(predictor))
    target(cell) = iszero(weight) ? zero(T) : -weight * inherited[cell]
    volume = workspace.cell_volume
    indices = workspace.pressure_indices
    # The volume defects divide by the open volume, at least the floor
    # `_MIMETIC_VOLUME_FLOOR h²` (`h²` without a solid).
    floored(c) = max(volume[c], T(_MIMETIC_VOLUME_FLOOR) * topology.spacing[c]^2)
    defect_of(outflow) = maximum(
        c -> volume[c] > zero(T) ? dt * abs(outflow[c] - target(c)) / floored(c) : zero(T), 1:cells,
    )
    multiplier = workspace.multiplier
    for cell in 1:cells
        workspace.anchor[cell] == cell && (multiplier[cell] = zero(T))
    end
    initial_residual = _mimetic_norm(_mimetic_outflow!(workspace.cell_scratch, workspace, predictor))
    _mimetic_recover!(workspace, multiplier)
    outflow = _mimetic_outflow!(workspace.cell_residual, workspace, workspace.velocity)
    reduced = length(indices)
    for vector in (
            workspace.reduced_rhs, workspace.reduced_solution, workspace.cg_residual,
            workspace.cg_direction, workspace.cg_preconditioned, workspace.cg_product,
            workspace.cg_schur, workspace.cg_scratch, workspace.cg_correction,
        )
        resize!(vector, reduced)
    end
    iterations = 0
    defect = defect_of(outflow)
    while defect > tolerance && iterations < max_iterations && reduced > 0
        # S δλ = target - outflow on the non-anchor cells.
        for (index, cell) in pairs(indices)
            workspace.reduced_rhs[index] = target(cell) - outflow[cell]
        end
        rhs_norm = _mimetic_norm(workspace.reduced_rhs)
        (isfinite(rhs_norm) && rhs_norm > zero(T)) || break
        requested = min(tolerance, tolerance / max(defect, eps(T)))
        attempt = _mimetic_pressure_cg!(
            workspace, preconditioner, requested, max_iterations - iterations,
        )
        correction = workspace.reduced_solution
        iterations += attempt
        (all(isfinite, correction) && attempt > 0) || break
        for (index, cell) in pairs(indices)
            multiplier[cell] += correction[index]
        end
        _mimetic_recover!(workspace, multiplier)
        outflow = _mimetic_outflow!(workspace.cell_residual, workspace, workspace.velocity)
        previous = defect
        defect = defect_of(outflow)
        defect < previous || break
    end
    final_residual = _mimetic_norm(outflow)
    # A plain loop: a closure over the reassigned `outflow` would box it.
    mixture_divergence = zero(T)
    for cell in 1:cells
        volume[cell] > zero(T) || continue
        mixture_divergence = max(mixture_divergence, abs(outflow[cell]) / floored(cell))
    end

    # Diagnostics in the metric 𝓜 and the transported momentum.
    projected = workspace.velocity
    mp = _mimetic_metric_product!(workspace.metric_predictor, workspace, predictor)
    mv = _mimetic_metric_product!(workspace.metric_velocity, workspace, projected)
    # The removed grid-line jump energy is 𝓜-orthogonal to the identified
    # space: it adds to E(u*) and to E(u* - u⁺) alike.
    energy_before = _mimetic_dot(predictor, mp) + removed_energy
    energy_after = _mimetic_dot(projected, mv)
    energy_correction = energy_before - 2 * _mimetic_dot(projected, mp) + energy_after
    identity = energy_before > zero(T) ?
        (energy_before - energy_after - energy_correction) / energy_before : zero(T)
    momentum = zeros(T, 2)
    wall = zeros(T, 2)
    transported = zeros(T, 2)
    transported_before = zeros(T, 2)
    for dof in eachindex(predictor)
        face = assembly.dof_face[dof]
        minus = topology.face_cells[1, face]
        axis = topology.sides[2, minus] == face || topology.second[2, minus] == face ? 1 : 2
        momentum[axis] += mv[dof] - mp[dof]
        # Wall coupling: the first moment of the constant field is 𝓜e_i + Ce_i.
        change = projected[dof] - predictor[dof]
        momentum[1] += _mimetic_wall_coupling(assembly, 1, dof) * change
        momentum[2] += _mimetic_wall_coupling(assembly, 2, dof) * change
        mass = assembly.dof_mass[dof]
        transported[axis] += mass * (projected[dof] - predictor[dof])
        transported_before[axis] += mass * predictor[dof]
        wall[axis] += topology.area[face] * assembly.dof_aperture[dof] * (
            multiplier[topology.face_cells[1, face]] -
                multiplier[topology.face_cells[2, face]]
        )
    end
    return (;
        iterations, initial_residual, final_residual, mixture_divergence,
        target_defect = defect, energy_identity_defect = identity,
        momentum_change = (momentum[1], momentum[2]),
        momentum_wall_force = (wall[1], wall[2]),
        transported_momentum_change = (transported[1], transported[2]),
        transported_momentum = (transported_before[1], transported_before[2]),
    )
end

"""Phase indicator of a cell for the capillary potentials: the Heaviside
`[C > ½]` (CGFM; `½` at `C = ½`, so that `I → 1 - I` when the labels are
exchanged) or the liquid fraction (CSF)."""
@inline _mimetic_indicator(formulation, fraction) = formulation isa CSF ? fraction :
    fraction > 0.5 ? one(fraction) : fraction == 0.5 ? one(fraction) / 2 : zero(fraction)

"""Cell curvature `κ_c` of the per-phase capillary potentials, in
`workspace.cell_curvature` (`NaN` where none is available).

A cut cell with a finite curvature keeps it: the height-function curvature
of [`reconstruct!`](@ref), in the first row along a domain wall with a contact
angle the energy-consistent contact-line curvature of
[`_contact_line_curvature!`](@ref). Every other cell, and a
cut cell without a finite curvature, takes the `C(1 - C)`-weighted mean of the
curvatures of its cut face neighbours; the curvatures of the cut cells are not
averaged among themselves. A pure cell can carry phase values of both phases
(the apertures of a face come from the adjacent PLIC traces) and then needs a
curvature: every cell whose potential enters a phase value with an aperture is
a face neighbour of a cut cell. A prescribed constant `curvature` fills every
cell."""
function _mimetic_cell_curvature!(
        workspace::MimeticProjectionWorkspace{T}, interface, curvature,
    ) where {T}
    references = workspace.references
    cells = length(references)
    kappa = resize!(workspace.cell_curvature, cells)
    if !isnothing(curvature)
        fill!(kappa, T(curvature))
        return kappa
    end
    weight = resize!(workspace.curvature_weight, cells)
    for cell in 1:cells
        reference = references[cell]
        fraction = _conditional_fraction(interface, reference)
        value = _get(interface.curvature, reference)
        if zero(T) < fraction < one(T) && isfinite(value)
            kappa[cell] = value
            weight[cell] = fraction * (one(T) - fraction)
        else
            kappa[cell] = T(NaN)
            weight[cell] = zero(T)
        end
    end
    topology = workspace.topology
    for cell in 1:cells
        iszero(weight[cell]) || continue
        total = weights = zero(T)
        for k in 1:4, face in (topology.sides[k, cell], topology.second[k, cell])
            face > 0 || continue
            minus = topology.face_cells[1, face]
            neighbour = minus == cell ? topology.face_cells[2, face] : minus
            w = weight[neighbour]
            if w > zero(T)
                total += w * kappa[neighbour]
                weights += w
            end
        end
        weights > zero(T) && (kappa[cell] = total / weights)
    end
    return kappa
end

"""Booked carriers `(b^l, b^g)` of the capillary difference term on a face
with both phase values (liquid aperture `liquid_aperture`, gas aperture
`gas_aperture`), for the current liquid and gas values in `velocity` and the
phase densities `density`.

For `Δt → 0` the VOF transport moves, per unit time, the liquid volume
`|f|(c^l u^l + c^g u^g)` through the face: the value of phase `π` moves the
liquid of its upstream cell `d(u^π)` (minus for `u ≥ 0`) that lies on its
subface `S^π`, `c^π = |S^π ∩ T^{d(u^π)}|` ([`_mimetic_carried_liquid`](@ref);
`T^d` the liquid trace of the PLIC line of `d` on the face, on a split side on
this face's segment, with a solid only its open part; `S^l`, `S^g` the
subfaces of widths `a^l`, `a^g` of the transport in the open part of the face,
the liquid one at the end of the averaged tangential PLIC normal,
[`_phase_subface_interval`](@ref)). The work `(b^l u^l + b^g u^g)(J^l - J^g)`
of the difference term is the surface-energy release `-σ Σ Δκ Φ` of the
transport while the clamp and the threshold below are inactive, where
`b^l u^l + b^g u^g = c^l u^l + c^g u^g`.

The phase roles follow the densities, not the labels: with the heavy phase
`H` (the liquid at equal densities), `λ = ρ_L/ρ_H` and `h^π` the volume of
`H` that value `π` carries (`c^π` if the liquid is heavy, `a^π - c^π`
otherwise), the light value keeps `λ h^L` and the heavy value books the rest
of the light value's share, `h^H + (1 - λ) h^L u^L/u^H`, clamped to
`[0, a^l + a^g]` (`h^H` for `|u^H| ≤ 10⁻¹⁰ |u^L|`, a heavy value negligible
against the light value, and for two zero values; the threshold is relative,
so the carriers are homogeneous of degree zero in the two values). At equal
densities each value keeps its own carrier, `b^π = c^π`; at a large density
ratio the light value gets almost no difference term, which would accelerate
it `ρ_H/ρ_L` times more than the heavy value. Exchanging the phase labels
exchanges `b^π` with `a^π - b^π`, so the capillary force is symmetric in the
labels ([`_mimetic_capillary_rhs!`](@ref))."""
@inline function _mimetic_booked_carriers(
        interface, entry, velocity, face, liquid_aperture::T, gas_aperture::T, density,
    ) where {T}
    ul = velocity[1].interior[face]
    ug = velocity[2].interior[face]
    ul = isfinite(ul) ? T(ul) : zero(T)
    ug = isfinite(ug) ? T(ug) : zero(T)
    liquid = _mimetic_carried_liquid(
        interface, entry, face, ul >= zero(T) ? entry.minus : entry.plus, true, liquid_aperture,
    )
    gas = _mimetic_carried_liquid(
        interface, entry, face, ug >= zero(T) ? entry.minus : entry.plus, false, gas_aperture,
    )
    total = liquid_aperture + gas_aperture
    liquid_density, gas_density = T(density[1]), T(density[2])
    gas_density > liquid_density ||
        return _mimetic_heavy_booking(liquid, gas, ul, ug, total, gas_density / liquid_density)
    # A heavy gas: the same booking of the gas volumes `a^π - c^π`.
    heavy, light = _mimetic_heavy_booking(
        gas_aperture - gas, liquid_aperture - liquid, ug, ul, total, liquid_density / gas_density,
    )
    return liquid_aperture - light, gas_aperture - heavy
end

"""Carriers `(b^H, b^L)` of the heavy-phase volume for the heavy and light
values, from the volumes `heavy` (`h^H`) and `light` (`h^L`) of the heavy
phase they carry, their values `uh` and `ul`, the open aperture `total` and
the density ratio `ratio = ρ_L/ρ_H` (see [`_mimetic_booked_carriers`](@ref))."""
@inline function _mimetic_heavy_booking(heavy::T, light::T, uh::T, ul::T, total::T, ratio::T) where {T}
    shifted = (one(T) - ratio) * light
    (iszero(shifted) || !(abs(uh) > T(1.0e-10) * abs(ul))) && return heavy, ratio * light
    return clamp(heavy + shifted * ul / uh, zero(T), total), ratio * light
end

"""Liquid of cell `donor` that the value of the liquid (`liquid_carrier`) or
gas phase of aperture `aperture` moves through `entry` (index `face`) per unit
velocity for `Δt → 0`: the overlap of its subface in the VOF transport
([`_phase_subface_interval`](@ref), current PLIC ordering) with the donor's
liquid trace on the face ([`_face_liquid_trace`](@ref)), relative to the face."""
@inline function _mimetic_carried_liquid(interface, entry, face, donor, liquid_carrier, aperture::T) where {T}
    geometry = interface.geometry
    lo, hi = _phase_subface_interval(interface, entry, face, donor, liquid_carrier, geometry, aperture, nothing)
    trace_lo, trace_hi, width = _face_liquid_trace(interface, geometry, entry, donor)
    return max(T(min(hi, trace_hi) - max(lo, trace_lo)), zero(T)) / T(width)
end

"""Interface point (offset from the cell centre in cell units) to which the
centred height-function curvature of cell `(i, j)` of the padded height
values refers: the mean, with the quality weights of `_lhf_geometry`, of the
column points `h₀ e_axis` of its valid candidates; `nothing` without one."""
function _mimetic_height_point(values, i, j)
    T = eltype(values)
    candidates = ntuple(_ -> (zero(T), zero(T), zero(T)), 4)
    count = 0
    for axis in 1:2, orientation in (-1, 1)
        heights = ntuple(3) do k
            offset = k - 2
            ii, jj = axis == 1 ? (i, j + offset) : (i + offset, j)
            _local_height(values, ii, jj, axis, orientation)
        end
        any(isnothing, heights) && continue
        hm, h0, hp = heights
        slope = (hp - hm) / 2
        quality = _height_quality(abs(orientation / sqrt(1 + slope^2)))
        quality > 0 || continue
        count += 1
        point = axis == 1 ? (T(h0), zero(T)) : (zero(T), T(h0))
        candidates = Base.setindex(candidates, (point..., T(quality)), count)
    end
    count == 0 && return nothing
    x = y = total = zero(T)
    for index in 1:count
        w = candidates[index][3]
        for other in 1:count
            other == index || (w *= one(T) - candidates[other][3])
        end
        x += w * candidates[index][1]
        y += w * candidates[index][2]
        total += w
    end
    total > zero(T) || return nothing
    return x / total, y / total
end

"""Point of the interface to which the curvature of cut cell `reference`
(size `h`) refers, on the cell's PLIC line: the point of its centred
height-function columns ([`_mimetic_height_point`](@ref)) projected onto the
PLIC line, or, where the centred columns are not valid (a generalised
height-function fit, a cell cut by a solid), the centroid of the PLIC line in
the cell, the origin of the fit (`_ghf_geometry`)."""
function _mimetic_curvature_point(interface, values, reference, h::T) where {T}
    center = cellcenter(interface.mesh, reference.block, reference.i, reference.j)
    mx, my = T(_get(interface.normal_x, reference)), T(_get(interface.normal_y, reference))
    intercept = T(_get(interface.intercept, reference))
    point = _mimetic_height_point(values[reference.block], reference.i + 4, reference.j + 4)
    norm = hypot(mx, my)
    if isnothing(point) || iszero(norm)
        cx, cy = _interface_centroid(mx, my, intercept)
        return center[1] + h * cx, center[2] + h * cy
    end
    nx, ny = mx / norm, my / norm
    # Distance of the column point from the PLIC line n ⋅ ξ = intercept/|n|.
    distance = point[1] * nx + point[2] * ny - intercept / norm
    return center[1] + h * (point[1] - distance * nx), center[2] + h * (point[2] - distance * ny)
end

"""Gravity components and density contrast `(g_x, g_y, ρ_l - ρ_g)` of the
hydrostatic part `(ρ_l - ρ_g) g ⋅ x*_c` of the potential difference
([`_mimetic_cell_potential!`](@ref)); `nothing` without gravity or surface
tension, where the potential difference has no hydrostatic part."""
@inline function _mimetic_hydrostatic(gravity, sigma, density, ::Type{T}) where {T}
    (isnothing(gravity) || all(iszero, gravity) || iszero(sigma)) && return nothing
    return T(gravity[1]), T(gravity[2]), T(density[1]) - T(density[2])
end

"""Hydrostatic part `(ρ_l - ρ_g) g ⋅ x*_c` of the potential difference of
`cell` (zero for `hydrostatic === nothing`, see [`_mimetic_hydrostatic`](@ref))."""
@inline function _mimetic_hydrostatic_potential(assembly, cell, hydrostatic)
    T = eltype(assembly.pressure_x)
    isnothing(hydrostatic) && return zero(T)
    gx, gy, contrast = hydrostatic
    return contrast * (gx * assembly.pressure_x[cell] + gy * assembly.pressure_y[cell])
end

"""Potential difference `ψ_c = φ^g_c - φ^l_c` of the per-phase capillary
potentials (`workspace.cell_potential`), from the cell curvatures `kappa` of
[`_mimetic_cell_curvature!`](@ref).

Without gravity (or with `σ = 0`) `ψ = σκ`. With gravity `g` one pressure per
cell balances the per-phase potentials and the Gram moment of gravity
(`ρ_π a|f| g ⋅ (x*₊ - x*₋)` on each value) exactly where
`Ψ_c = ψ_c - (ρ_l - ρ_g) g ⋅ x*_c` is the same on the cells that carry values
of both phases. In the continuum the jump at a point `x` off the interface is
the hydrostatic extension `σκ(x_Γ) + (ρ_l - ρ_g) g ⋅ (x - x_Γ)` from the point
`x_Γ` to which `κ` refers. So a cell with its own curvature (a cut cell with a
finite curvature; every cut cell for a `prescribed` curvature) has
`Ψ_c = σκ_c - (ρ_l - ρ_g) g ⋅ x_Γ` with `x_Γ` its curvature point on the PLIC
line ([`_mimetic_curvature_point`](@ref)), every other cell the
`C(1 - C)`-weighted mean of the `Ψ` of its cut face neighbours (on a split
side both faces), and `ψ_c = Ψ_c + (ρ_l - ρ_g) g ⋅ x*_c`. A flat interface
normal to `g` (`κ = 0`, every `x_Γ` on one level line) stays at rest exactly."""
function _mimetic_cell_potential!(
        workspace::MimeticProjectionWorkspace{T}, interface, kappa, sigma, gravity, density;
        prescribed = false,
    ) where {T}
    cells = length(kappa)
    psi = resize!(workspace.cell_potential, cells)
    hydrostatic = _mimetic_hydrostatic(gravity, sigma, density, T)
    if isnothing(hydrostatic)
        for cell in 1:cells
            psi[cell] = sigma * kappa[cell]
        end
        return psi
    end
    values = _fill_height_values!(interface)
    assembly = workspace.assembly
    topology = workspace.topology
    references = workspace.references
    gx, gy, contrast = hydrostatic
    # The weights of `_mimetic_cell_curvature!` mark its cells with own curvature.
    weight = resize!(workspace.curvature_weight, cells)
    for cell in 1:cells
        reference = references[cell]
        fraction = _conditional_fraction(interface, reference)
        if zero(T) < fraction < one(T) && (prescribed || weight[cell] > zero(T))
            weight[cell] = fraction * (one(T) - fraction)
            x, y = _mimetic_curvature_point(interface, values, reference, topology.spacing[cell])
            psi[cell] = sigma * kappa[cell] - contrast * (gx * x + gy * y)
        else
            weight[cell] = zero(T)
            psi[cell] = T(NaN)
        end
    end
    for cell in 1:cells
        iszero(weight[cell]) || continue
        total = weights = zero(T)
        for k in 1:4, face in (topology.sides[k, cell], topology.second[k, cell])
            face > 0 || continue
            minus = topology.face_cells[1, face]
            neighbour = minus == cell ? topology.face_cells[2, face] : minus
            w = weight[neighbour]
            if w > zero(T)
                total += w * psi[neighbour]
                weights += w
            end
        end
        psi[cell] = weights > zero(T) ? total / weights : T(NaN)
    end
    for cell in 1:cells
        psi[cell] += _mimetic_hydrostatic_potential(assembly, cell, hydrostatic)
    end
    return psi
end

"""Add the per-phase surface-tension increment `dt 𝓜⁻¹ f^π` to each phase,
with the right-hand side `f` of [`_mimetic_capillary_rhs!`](@ref) for the
potential difference `ψ` of [`_mimetic_cell_potential!`](@ref) (`σκ` with the
cell curvature `κ` of [`_mimetic_cell_curvature!`](@ref); with `gravity`
hydrostatically referred to the curvature's point). A face whose force needs
a curvature that is not available gets no force; such faces are counted in
`unresolved_curvature_faces`. `curvature` prescribes a constant curvature
(tests of the balance)."""
function _mimetic_capillary_force!(
        velocity::TwoPhaseVelocity, solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, dt;
        sigma, formulation = CGFM(), curvature = nothing, gravity = nothing,
        unresolved_curvature_faces::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {T}
    _validate_surface_tension_formulation(formulation)
    isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
    isnothing(curvature) || isfinite(curvature) || throw(ArgumentError("constant curvature must be finite"))
    interface = solver.interface
    assembly = _mimetic_assemble!(solver, workspace)
    count = length(assembly.band)
    resize!(workspace.band_scratch, count)
    resize!(workspace.band_solution, count)
    kappa = _mimetic_cell_curvature!(workspace, interface, curvature)
    sigma = T(sigma)
    potential = _mimetic_cell_potential!(
        workspace, interface, kappa, sigma, gravity, solver.density; prescribed = !isnothing(curvature),
    )
    rhs, unresolved = _mimetic_capillary_rhs!(
        workspace, solver, velocity, potential; sigma, formulation, curvature, gravity,
    )
    isnothing(unresolved_curvature_faces) || (unresolved_curvature_faces[] = unresolved)
    acceleration = _mimetic_metric_solve!(workspace.force_solution, workspace, rhs)
    for dof in eachindex(acceleration)
        _mimetic_scatter!(velocity, assembly, dof, T(dt) * acceleration[dof], true)
    end
    return velocity
end

"""Right-hand side `f^π` of the per-phase capillary force
(`workspace.force_rhs`) for the cell potential difference `potential` (`ψ`)
of the assembled metric, and the number of faces without force.

Each phase has its own capillary potential, `φ^g = ψI` and `φ^l = ψI - ψ`,
with the indicator `I` (Heaviside `[C > ½]` for CGFM, the liquid fraction for
CSF), and the value of phase `π` on face `f` gets the integrated jump
`J^π_f = |f| (φ^π_+ - φ^π_-)`. The two potentials differ by `ψ`, the
capillary pressure jump, so a static drop of constant curvature is exactly
balanced (both jumps are then the common CGFM/CSF jump `σκ [I]_f |f|`, a
gradient), while a varying curvature drives the values with their booked
carriers (up to a common gradient the force is `-b^π |f| Δ_f ψ` on value `π`,
almost only on the heavy value at a large density ratio). A value alone on its
face gets `f^π = a^π J^π`. On a face with both values the value of phase `π`
gets `f^π = a^π J^g + b^π (J^l - J^g - H_f) + [π = l] a^l H_f`, where
`J^l - J^g = -|f| Δ_f ψ` and `H_f = -|f| Δ_f ((ρ_l - ρ_g) g ⋅ x*)` is its
hydrostatic part (zero without gravity or surface tension, see
[`_mimetic_hydrostatic`](@ref)): the capillary part `-|f| Δ_f Ψ` rides on the
booked carriers `b^π` of [`_mimetic_booked_carriers`](@ref), which book the
liquid volume the VOF transport moves through the face, so that its work is
the surface-energy release of the transport; the hydrostatic part keeps the
liquid aperture, the carrier of gravity's Gram moment, so that a hydrostatic
potential (`Ψ` constant) is balanced by one pressure per cell at any carrier.
Exchanging the phase labels exchanges the potentials (`ψ → -ψ`, `I → 1 - I`
away from `C = ½`), negates `H_f` and exchanges `b^π` with `a^π - b^π`, which
gives every value the same `f^π`: the force is symmetric in the labels. An
identified grid-line value (one value for both phases on a face between a
pure liquid and a pure gas cell) gets the common jump `σ κ_f [I]_f |f|` with
the face curvature of [`surface_tension!`](@ref) (or the prescribed
`curvature`)."""
function _mimetic_capillary_rhs!(
        workspace::MimeticProjectionWorkspace{T}, solver::MimeticProjectionSolver{T}, velocity, potential;
        sigma, formulation, curvature, gravity,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    assembly = workspace.assembly
    topology = workspace.topology
    rhs = workspace.force_rhs
    resize!(rhs, length(assembly.dof_face))
    fill!(rhs, zero(T))
    sigma = T(sigma)
    hydrostatic = _mimetic_hydrostatic(gravity, sigma, solver.density, T)
    unresolved = 0
    for (face, entry) in pairs(mesh.faces)
        liquid, gas = assembly.liquid_dof[face], assembly.gas_dof[face]
        minus, plus = topology.face_cells[1, face], topology.face_cells[2, face]
        cm = _conditional_fraction(interface, entry.minus)
        cp = _conditional_fraction(interface, entry.plus)
        if liquid > 0 && liquid == gas
            jump = _surface_tension_jump(formulation, cm, cp)
            (iszero(sigma) || iszero(jump)) && continue
            value = isnothing(curvature) ? _face_curvature(interface, entry) : curvature
            if isnothing(value)
                unresolved += 1
            else
                rhs[liquid] = sigma * value * jump * entry.area
            end
            continue
        end
        im, ip = _mimetic_indicator(formulation, cm), _mimetic_indicator(formulation, cp)
        # Potentials φ^g = ψI and φ^l = -ψ(1 - I) (ψ = σκ without gravity); a
        # vanishing factor needs no curvature.
        gas_aperture = gas > 0 ? assembly.dof_aperture[gas] : zero(T)
        liquid_aperture = liquid > 0 ? assembly.dof_aperture[liquid] : zero(T)
        gas_minus = iszero(im) ? zero(T) : potential[minus] * im
        gas_plus = iszero(ip) ? zero(T) : potential[plus] * ip
        liquid_minus = isone(im) ? zero(T) : -potential[minus] * (one(T) - im)
        liquid_plus = isone(ip) ? zero(T) : -potential[plus] * (one(T) - ip)
        gas_jump = entry.area * (gas_plus - gas_minus)
        liquid_jump = entry.area * (liquid_plus - liquid_minus)
        if (gas_aperture > zero(T) && !isfinite(gas_jump)) ||
                (liquid_aperture > zero(T) && !isfinite(liquid_jump))
            unresolved += 1
            continue
        end
        gas_aperture > zero(T) && (rhs[gas] = gas_aperture * gas_jump)
        liquid_aperture > zero(T) || continue
        if gas_aperture > zero(T)
            # Both values: the capillary part of the difference term rides on
            # the booked carriers of the two values, its hydrostatic part on
            # the liquid aperture.
            liquid_carrier, gas_carrier = _mimetic_booked_carriers(
                interface, entry, velocity, face, liquid_aperture, gas_aperture, solver.density,
            )
            hydrostatic_jump = -entry.area * (
                _mimetic_hydrostatic_potential(assembly, plus, hydrostatic) -
                    _mimetic_hydrostatic_potential(assembly, minus, hydrostatic)
            )
            difference = liquid_jump - gas_jump - hydrostatic_jump
            rhs[gas] += gas_carrier * difference
            rhs[liquid] = liquid_aperture * gas_jump + liquid_carrier * difference +
                liquid_aperture * hydrostatic_jump
        else
            rhs[liquid] = liquid_aperture * liquid_jump
        end
    end
    return rhs, unresolved
end

"""`y = (𝓜 + half (RᵀKR + A_drag)) x`, matrix-free (`half < 0` gives the
right-hand side). `R x` is the staggered-mass mean of the values of a face,
`Rᵀ` hands a face value back to the phase values with the weights `m_d / B_f`,
`K` is applied in its factorized form `EᵀW_μE` and `A_drag` face by face with
the coefficients `work.drag`."""
function _mimetic_viscous_apply!(y, workspace, operator, half, x)
    work = workspace.viscous
    assembly = workspace.assembly
    _mimetic_metric_product!(y, workspace, x)
    mean = work.face_sum
    mass = work.face_mass
    for face in eachindex(mean)
        liquid = assembly.liquid_dof[face]
        gas = assembly.gas_dof[face]
        # A face without open volume (covered by a solid) has no value: the
        # solid's velocity, zero.
        mean[face] = if liquid == 0 && gas == 0
            zero(eltype(x))
        elseif liquid == gas || gas == 0
            x[liquid]
        elseif liquid == 0
            x[gas]
        else
            muladd(
                assembly.dof_mass[liquid], x[liquid], assembly.dof_mass[gas] * x[gas],
            ) / mass[face]
        end
    end
    stress = work.face_stress
    _factorized_variable_stiffness_mul!(stress, operator, mean, work.stress_work)
    for dof in eachindex(y)
        face = assembly.dof_face[dof]
        y[dof] = muladd(half * assembly.dof_mass[dof] / mass[face], stress[face], y[dof])
    end
    drag = work.drag
    for face in eachindex(drag)
        coefficient = drag[face]
        iszero(coefficient) && continue
        liquid = assembly.liquid_dof[face]
        gas = assembly.gas_dof[face]
        force = half * coefficient * (x[liquid] - x[gas])
        y[liquid] += force
        y[gas] -= force
    end
    return y
end

"""`y ≈ (B + half K)⁻¹ x` on the faces: one cycle of the mixture solve's
preconditioner (refreshed for the staggered masses `B` of the metric), or the
inverse diagonal of its matrix if it has none."""
function _mimetic_mixture_cycle!(y, work, mixture_solver, x)
    preconditioner = mixture_solver.preconditioner
    isnothing(preconditioner) || return ldiv!(y, preconditioner, x)
    y .= x ./ work.face_diagonal
    return y
end

"""Lifted mixture preconditioner of the viscous step for `𝓜 = W`,

    z = (G⁻¹ - P B⁻¹ Pᵀ) r + P 𝒱 Pᵀ r,

with the face-local `G = W + half A_drag` (the staggered phase masses `W` and
the drag of the face), `P` the copy of a face value to its phase values,
`B = PᵀWP` and `𝒱 ≈ (B + half K)⁻¹` (`_mimetic_mixture_cycle!`). Since
`GP = WP`, Woodbury gives `(G + RᵀKR)⁻¹ = G⁻¹ - PB⁻¹Pᵀ + P(B + half K)⁻¹Pᵀ`
(`R = B⁻¹PᵀW`): for `𝓜 = W` and an exact `𝒱` this is the inverse of the
system. `G⁻¹ - PB⁻¹Pᵀ = G^{-1/2}(I - Π)G^{-1/2}` with the `G`-orthogonal
projector `Π` onto the face fields is positive semi-definite, so `z` is
symmetric positive definite with `𝒱`."""
function _mimetic_lifted_mixture!(z, workspace, mixture_solver, half, r)
    work = workspace.viscous
    assembly = workspace.assembly
    sum = work.face_sum
    for face in eachindex(sum)
        liquid = assembly.liquid_dof[face]
        gas = assembly.gas_dof[face]
        sum[face] = liquid == 0 && gas == 0 ? zero(eltype(r)) :
            liquid == gas || gas == 0 ? r[liquid] :
            liquid == 0 ? r[gas] : r[liquid] + r[gas]
    end
    correction = _mimetic_mixture_cycle!(work.face_correction, work, mixture_solver, sum)
    mass = work.face_mass
    drag = work.drag
    for face in eachindex(sum)
        liquid = assembly.liquid_dof[face]
        gas = assembly.gas_dof[face]
        if liquid == 0 && gas == 0
            continue
        elseif liquid == gas || gas == 0
            z[liquid] = correction[face]
        elseif liquid == 0
            z[gas] = correction[face]
        else
            liquid_mass = assembly.dof_mass[liquid]
            gas_mass = assembly.dof_mass[gas]
            coupling = half * drag[face]
            determinant = muladd(coupling, liquid_mass + gas_mass, liquid_mass * gas_mass)
            shift = correction[face] - sum[face] / mass[face]
            z[liquid] = muladd(gas_mass + coupling, r[liquid], coupling * r[gas]) /
                determinant + shift
            z[gas] = muladd(liquid_mass + coupling, r[gas], coupling * r[liquid]) /
                determinant + shift
        end
    end
    return z
end

"""Assemble the band block `A_bb = 𝓜_bb + half (R_bᵀKR_b + A_drag,bb)` of the
viscous operator on the band values (upper triangle; `K` assembled, the drag
from `work.drag`) and factorise it (sparse LDLᵀ, simplicial, without BLAS).
The symbolic analysis is reused while the sparsity pattern is unchanged."""
function _mimetic_viscous_band_factor!(workspace, operator, half)
    work = workspace.viscous
    assembly = workspace.assembly
    band = assembly.band
    position = assembly.band_position
    T = eltype(work.band_values)
    rows, columns, values = work.band_rows, work.band_columns, work.band_values
    empty!(rows)
    empty!(columns)
    empty!(values)
    metric = assembly.band_matrix
    for column in axes(metric, 2), pointer in nzrange(metric, column)
        row = rowvals(metric)[pointer]
        row <= column || continue
        push!(rows, row)
        push!(columns, column)
        push!(values, nonzeros(metric)[pointer])
    end
    stiffness = operator.stiffness
    mass = work.face_mass
    for (column, dof) in pairs(band)
        face = assembly.dof_face[dof]
        weight = half * assembly.dof_mass[dof] / mass[face]
        for pointer in nzrange(stiffness, face)
            other = rowvals(stiffness)[pointer]
            liquid = assembly.liquid_dof[other]
            gas = assembly.gas_dof[other]
            for target in (liquid == gas ? (liquid, 0) : (liquid, gas))
                target > 0 || continue
                row = position[target]
                0 < row <= column || continue
                push!(rows, row)
                push!(columns, column)
                push!(values, weight * nonzeros(stiffness)[pointer] * assembly.dof_mass[target] / mass[other])
            end
        end
    end
    for face in eachindex(work.drag)
        coefficient = half * work.drag[face]
        iszero(coefficient) && continue
        liquid = position[assembly.liquid_dof[face]]
        gas = position[assembly.gas_dof[face]]
        for row in (liquid, gas)
            row > 0 || continue
            push!(rows, row)
            push!(columns, row)
            push!(values, coefficient)
        end
        if liquid > 0 && gas > 0
            push!(rows, min(liquid, gas))
            push!(columns, max(liquid, gas))
            push!(values, -coefficient)
        end
    end
    count = length(band)
    resize!(work.band_rhs, count)
    resize!(work.band_solution, count)
    return _mimetic_ldlt!(work.band_ldlt, Symmetric(sparse(rows, columns, values, count, count), :U))
end

"""`z = E A_bb⁻¹ Eᵀ r` (`add = false`, zero off the band) or `z += E A_bb⁻¹ Eᵀ r`."""
function _mimetic_band_correction!(z, workspace, r, add)
    work = workspace.viscous
    band = workspace.assembly.band
    add || fill!(z, zero(eltype(z)))
    for (position, dof) in pairs(band)
        work.band_rhs[position] = r[dof]
    end
    _mimetic_ldlt_solve!(work.band_solution, work.band_ldlt, work.band_rhs)
    for (position, dof) in pairs(band)
        z[dof] += work.band_solution[position]
    end
    return z
end

"""Preconditioner of the viscous step: the lifted mixture preconditioner `L`
(`_mimetic_lifted_mixture!`, exact for `𝓜 = W`) inside a symmetric
multiplicative correction with the exact band block `C = E A_bb⁻¹ Eᵀ`
(`_mimetic_viscous_band_factor!`), where `𝓜 ≠ W`,

    z₁ = C r,   z₂ = z₁ + L (r - A z₁),   z = z₂ + C (r - A z₂).

`I - CA` is the `A`-orthogonal projector `Q` onto the complement of the band
values, the error propagation is `Q(I - LA)Q`, and `rᵀz = |(I - Q)A⁻¹r|²_A +
(AQA⁻¹r)ᵀL(AQA⁻¹r) > 0`: symmetric positive definite for every symmetric
positive definite `L`, independent of the time step. Without a band, or with
`lifted_only`, it is `L`."""
function _mimetic_viscous_precondition!(
        z, workspace, operator, mixture_solver, half, r; lifted_only = false,
    )
    band = workspace.assembly.band
    if lifted_only || isempty(band)
        return _mimetic_lifted_mixture!(z, workspace, mixture_solver, half, r)
    end
    work = workspace.viscous
    product, residual = work.band_product, work.band_residual
    _mimetic_band_correction!(z, workspace, r, false)
    _mimetic_viscous_apply!(product, workspace, operator, half, z)
    residual .= r .- product
    _mimetic_lifted_mixture!(product, workspace, mixture_solver, half, residual)
    z .+= product
    _mimetic_viscous_apply!(product, workspace, operator, half, z)
    residual .= r .- product
    return _mimetic_band_correction!(z, workspace, residual, true)
end

"""Serial (thread- and BLAS-independent) dot product."""
function _mimetic_dot(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    sum = zero(T)
    for index in eachindex(a, b)
        sum = muladd(a[index], b[index], sum)
    end
    return sum
end

"""Serial Euclidean norm."""
_mimetic_norm(a::AbstractVector) = sqrt(_mimetic_dot(a, a))

"""Serial maximum norm."""
function _mimetic_max_norm(a::AbstractVector{T}) where {T}
    result = zero(T)
    for value in a
        result = max(result, abs(value))
    end
    return result
end

"""Preconditioned conjugate gradients for the viscous system from
`x = work.values`, stopped when the max-norm residual is at most `ratio`
times that of the right-hand side; restarted from the true residual if the
recursive one drifted below the target. Reductions are serial, so the bits
do not depend on the thread counts. If round-off makes `rᵀz` or `pᵀAp`
non-positive, the iteration restarts from the true residual with the lifted
preconditioner alone (counted in `work.restarts`). Returns the iterations and
the relative true residual."""
function _mimetic_viscous_cg!(workspace, operator, mixture_solver, half, ratio, max_cycles)
    work = workspace.viscous
    x, b, r = work.solution, work.rhs, work.residual
    z, p, q = work.preconditioned, work.direction, work.product
    T = eltype(x)
    copyto!(x, work.values)
    scale = max(_mimetic_max_norm(b), floatmin(T))
    target = ratio * scale
    _mimetic_viscous_apply!(q, workspace, operator, half, x)
    r .= b .- q
    residual = _mimetic_max_norm(r)
    iterations = 0
    lifted_only = false
    while residual > target && iterations < max_cycles
        _mimetic_viscous_precondition!(z, workspace, operator, mixture_solver, half, r; lifted_only)
        rz = _mimetic_dot(r, z)
        copyto!(p, z)
        breakdown = !(rz > zero(T))
        while !breakdown && iterations < max_cycles
            _mimetic_viscous_apply!(q, workspace, operator, half, p)
            curvature = _mimetic_dot(p, q)
            breakdown = !(curvature > zero(T))
            breakdown && break
            alpha = rz / curvature
            x .+= alpha .* p
            r .-= alpha .* q
            iterations += 1
            _mimetic_max_norm(r) <= target && break
            _mimetic_viscous_precondition!(z, workspace, operator, mixture_solver, half, r; lifted_only)
            next = _mimetic_dot(r, z)
            breakdown = !(next > zero(T))
            breakdown && break
            p .= z .+ (next / rz) .* p
            rz = next
        end
        if breakdown
            lifted_only && throw(
                ErrorException("mimetic viscous solve: conjugate gradients broke down with the lifted preconditioner"),
            )
            lifted_only = true
            work.restarts += 1
        end
        _mimetic_viscous_apply!(q, workspace, operator, half, x)
        r .= b .- q
        residual = _mimetic_max_norm(r)
    end
    return iterations, residual / scale
end

"""Implicit-midpoint viscous step of the phase values in the metric `𝓜`.

With the mixture map `R = B⁻¹PᵀW` (the staggered-mass mean `ū = Ru` of the
values of a face, `B = PᵀWP`), the symmetric positive semi-definite stress
operator `K` of the mixture solve (`operator`) and the interphase drag
`A_drag` of `drag` (`_interphase_drag_coefficients!`, zero for
`drag = nothing`), `A = RᵀKR + A_drag` and

    (𝓜 + dt/2 A) u⁺ = (𝓜 - dt/2 A) u.

The amplification factors `(1 - dtλ/2)/(1 + dtλ/2)` of the pencil `(A, 𝓜)` lie
in `(-1, 1]` for every `dt`: the step does not increase `uᵀ𝓜u`, the energy of
the projection, for any time step. For `𝓜 = W` and no drag it is the common
increment of the Crank--Nicolson mixture solve, `u⁺ = u + Pδ`. The values of
`source` are gathered as for the projection (an identified grid-line value is
the mass-weighted mean of the two phase values) and the result is written to
`destination`; absent (`NaN`) values stay absent. The system is solved
matrix-free by conjugate gradients (`_mimetic_viscous_cg!`) with the
preconditioner `_mimetic_viscous_precondition!` (the lifted mixture
preconditioner, whose cycle is the mixture solve's own, refreshed here for the
masses `B` and required to be symmetric, inside an exact band-block
correction), to the relative max-norm residual `tolerance`; returns that
residual, the iterations and the number of unknowns."""
function _mimetic_viscous_step!(
        destination::TwoPhaseVelocity, source::TwoPhaseVelocity,
        solver::MimeticProjectionSolver{T}, workspace::MimeticProjectionWorkspace{T},
        operator, mixture_solver, dt, drag, model;
        tolerance, max_cycles,
    ) where {T}
    assembly = _mimetic_assemble!(solver, workspace)
    work = workspace.viscous
    dofs = length(assembly.dof_face)
    faces = length(operator.volumes)
    _mimetic_gather!(work.values, assembly, source)
    for vector in (
            work.solution, work.rhs, work.residual, work.direction,
            work.preconditioned, work.product, work.band_product, work.band_residual,
        )
        resize!(vector, dofs)
    end
    for vector in (
            work.face_mass, work.face_density, work.face_sum, work.face_stress,
            work.face_correction, work.face_diagonal, work.drag,
        )
        resize!(vector, faces)
    end
    resize!(work.stress_work, length(operator.stress_weights))
    for face in 1:faces
        liquid = assembly.liquid_dof[face]
        gas = assembly.gas_dof[face]
        mass = liquid == 0 && gas == 0 ? zero(T) :
            liquid == gas || gas == 0 ? assembly.dof_mass[liquid] :
            liquid == 0 ? assembly.dof_mass[gas] :
            assembly.dof_mass[liquid] + assembly.dof_mass[gas]
        work.face_mass[face] = mass
        # A face without values (no open volume) keeps the mixture density of
        # the common viscous solve.
        work.face_density[face] = iszero(mass) ? workspace.face_data.total_mass[face] :
            mass / operator.volumes[face]
    end
    _interphase_drag_coefficients!(
        work.drag, assembly, work.values, drag, model, operator.averaging,
    )
    half = T(dt) / 2
    _update_variable_diffusion_solver!(mixture_solver, operator, work.face_density, T(dt))
    # The lifted preconditioner is symmetric only for a symmetric cycle.
    _validate_cg_preconditioner(mixture_solver.preconditioner)
    isempty(assembly.band) || _mimetic_viscous_band_factor!(workspace, operator, half)
    if isnothing(mixture_solver.preconditioner)
        matrix = mixture_solver.matrix
        rows = rowvals(matrix)
        for face in 1:faces, pointer in nzrange(matrix, face)
            rows[pointer] == face && (work.face_diagonal[face] = nonzeros(matrix)[pointer])
        end
    end
    _mimetic_viscous_apply!(work.rhs, workspace, operator, -half, work.values)
    iterations, residual = _mimetic_viscous_cg!(
        workspace, operator, mixture_solver, half,
        _dimensionless_tolerance(T, tolerance), max_cycles,
    )
    _copy_velocity!(destination, source)
    for dof in 1:dofs
        _mimetic_scatter!(destination, assembly, dof, work.solution[dof], false)
    end
    return (; residual, iterations, unknowns = dofs)
end

"""Add the gravity increment to every active interior phase value: the uniform
acceleration `dt g` of both phases, plus `dt 𝓜⁻¹(C g)` in the cut cells at a
wall and on values tied to a wall (`C`, see `_mimetic_wall_coupling`).

The body force is the Gram moment `R g` of the constant field `g`, the work
`ρ_π g ⋅ u` of the cut-cell blocks. Away from walls the metric reproduces it,
`𝓜 g = R g`, so the increment is `dt g`. A wall side has a prescribed flux, not
a value, and the block is restricted to the interior sides; there
`R g = 𝓜 g + C g` with the coupling `C` of the values to the wall flux density
`n_w ⋅ g`, which vanishes on walls parallel to `g`. On every value `R g` is the
face difference `ρ_π a|f| (x*₊ - x*₋) ⋅ g` of the pressure points, so a fluid
whose cut-cell pressure points lie on a level line is at rest. A density-weighted
jump force `𝓜⁻¹(a^π J)` would instead give the light phase value of a mixed
face the weight of the mixture."""
function _mimetic_uniform_acceleration!(
        velocity::TwoPhaseVelocity, solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, gravity, dt,
    ) where {T}
    mesh = solver.interface.mesh
    all(isfinite, gravity) || throw(ArgumentError("gravity must be finite"))
    any(axis -> mesh.periodic[axis] && !iszero(gravity[axis]), 1:2) && throw(
        ArgumentError("gravity potential requires nonperiodic boundaries along gravity"),
    )
    for phase in 1:2
        values = velocity[phase].interior
        for (index, face) in pairs(mesh.faces)
            isnan(values[index]) || (values[index] += dt * gravity[face.axis])
        end
    end
    assembly = _mimetic_assemble!(solver, workspace)
    any(!iszero, assembly.wall_coupling) || return velocity
    rhs = resize!(workspace.force_rhs, length(assembly.dof_face))
    for dof in eachindex(rhs)
        rhs[dof] = _mimetic_wall_coupling(assembly, 1, dof) * T(gravity[1]) +
            _mimetic_wall_coupling(assembly, 2, dof) * T(gravity[2])
    end
    count = length(assembly.band)
    resize!(workspace.band_scratch, count)
    resize!(workspace.band_solution, count)
    acceleration = _mimetic_metric_solve!(workspace.force_solution, workspace, rhs)
    for dof in eachindex(acceleration)
        _mimetic_scatter!(velocity, assembly, dof, T(dt) * acceleration[dof], true)
    end
    return velocity
end
