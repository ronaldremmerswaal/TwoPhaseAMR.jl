# Kinetic-energy-orthogonal two-velocity projection in the mimetic cut-cell
# metric (`MimeticProjection`). Uniform square-cell meshes without embedded
# solids only. See docs/techref/chapters/10_projection.tex, "Mimetic cut-cell
# projection".

import SparseArrays: UMFPACK

"""
    MimeticProjectionSolver(interface; density=(1, 1))

Configuration of the mimetic cut-cell projection on `interface`. Throws
`ArgumentError` for embedded solids, non-uniform (adaptive) meshes, non-square
cells and element types other than `Float64`.
"""
struct MimeticProjectionSolver{T, I}
    interface::I
    density::NTuple{2, T}
    allow_wall_cuts::Bool
end

function MimeticProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
        _allow_wall_cuts = false,
    ) where {T}
    T === Float64 || throw(ArgumentError("MimeticProjection requires Float64"))
    rho = (T(density[1]), T(density[2]))
    all(value -> isfinite(value) && value > zero(T), rho) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    isnothing(interface.geometry) || throw(
        ArgumentError("MimeticProjection does not support embedded solid geometry"),
    )
    _mimetic_uniform_spacing(interface.mesh)
    return MimeticProjectionSolver{T, typeof(interface)}(
        interface, rho, Bool(_allow_wall_cuts),
    )
end

"""Gauge-reduced Schur complement `D 𝓜⁻¹ Dᵀ` of the mimetic saddle-point system.

Applied matrix-free: the band block of `𝓜` is solved with its sparse
factorization, the diagonal part of `𝓜` enters through an assembled
cell matrix. Cell 1 is the pressure anchor.
"""
struct _MimeticSchurOperator{T, W}
    workspace::W
    size::Int
end

Base.size(op::_MimeticSchurOperator) = (op.size, op.size)
Base.size(op::_MimeticSchurOperator, dimension::Integer) = op.size
Base.eltype(::_MimeticSchurOperator{T}) where {T} = T

"""Reusable fields and the most recently assembled mimetic system."""
mutable struct MimeticProjectionWorkspace{T, M <: BlockTreeMesh}
    liquid_fraction_x::FaceField{1, T, M}
    liquid_fraction_y::FaceField{2, T, M}
    face_data::Union{Nothing, _ProjectionFaceData{T}}
    topology::Union{Nothing, _MimeticTopology{T}}
    references::Vector{CellRef}
    assembly::_MimeticAssembly{T}
    factor::Union{Nothing, UMFPACK.UmfpackLU{Float64, Int}}
    divergence_band::SparseMatrixCSC{T, Int}
    divergence_band_transpose::SparseMatrixCSC{T, Int}
    diagonal_laplacian::SparseMatrixCSC{T, Int}
    boundary_flux::Vector{T}
    boundary_aperture::Vector{T}
    boundary_low::Vector{Bool}
    gridline::Vector{Int8}
    preconditioner_coefficient::Vector{T}
    pressure_operator::Any
    multigrid::Any
    preconditioner::Any
    predictor::Vector{T}
    velocity::Vector{T}
    metric_predictor::Vector{T}
    metric_velocity::Vector{T}
    multiplier::Vector{T}
    cell_residual::Vector{T}
    cell_scratch::Vector{T}
    band_scratch::Vector{T}
    band_solution::Vector{T}
    reduced_rhs::Vector{T}
    reduced_solution::Vector{T}
    krylov::Union{Nothing, Krylov.CgWorkspace{T, T, Vector{T}}}
    force_rhs::Vector{T}
    force_solution::Vector{T}
    cell_curvature::Vector{T}
    curvature_weight::Vector{T}
    viscous_values::Vector{T}
    viscous_face_mass::Vector{T}
end

"""
    MimeticProjectionWorkspace(solver::MimeticProjectionSolver)

Allocate reusable storage for [`project!`](@ref) with a mimetic projection.
"""
function MimeticProjectionWorkspace(solver::MimeticProjectionSolver{T}) where {T}
    mesh = solver.interface.mesh
    return MimeticProjectionWorkspace{T, typeof(mesh)}(
        FaceField{1}(mesh, T), FaceField{2}(mesh, T), nothing, nothing,
        CellRef[], _MimeticAssembly(T), nothing, spzeros(T, 0, 0), spzeros(T, 0, 0),
        spzeros(T, 0, 0), T[], T[], Bool[], Int8[], T[], nothing, nothing, nothing,
        T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], T[], nothing, T[], T[], T[], T[], T[], T[],
    )
end

"""Diagnostics of one mimetic cut-cell projection.

`mixture_volume_defect` is `dt * max(abs(div(u_mix)))` of the projected state;
`target_defect` the same relative to an inherited mixture-flux target.
`energy_identity_defect` is `(E(u*) - E(u⁺) - E(u* - u⁺)) / E(u*)` in the norm
of the metric `𝓜`; on grid-line interface faces `E(u*)` and `E(u* - u⁺)`
include the energy of the phase-velocity jump that identifying the two phase
values removes. `momentum_change` is the change of the first-moment
momentum `e_iᵀ𝓜u` and `momentum_wall_force` the wall term `dt (D e_i)ᵀ p` it
must equal; `transported_momentum_change` is the change of the transported
momentum `e_iᵀWu` with the diagonal phase masses `W`, which this projection
does not conserve, and `transported_momentum` its value before projection.
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
        ", slaved_values=", stats.slaved_values,
        ", stabilised_values=", stats.stabilised_values,
        ", lumped_values=", stats.lumped_values,
        ", defective_cells=", stats.defective_cells,
        ", gridline_faces=", stats.gridline_faces, ")",
    )
end

"""Assemble the metric and the reduced operators for the current interface.

The assembly is reused while the interface fractions (hence apertures) and
the mesh topology are unchanged.
"""
function _mimetic_assemble!(
        solver::MimeticProjectionSolver{T}, workspace::MimeticProjectionWorkspace{T},
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    topology_changed = !_mimetic_topology_current(workspace.topology, mesh)
    if topology_changed
        workspace.topology = _mimetic_topology(mesh)
        workspace.references = _cell_references(mesh)
        _resize_face_field!(workspace.liquid_fraction_x, mesh)
        _resize_face_field!(workspace.liquid_fraction_y, mesh)
    end
    data = _projection_face_data(solver, workspace; storage = workspace.face_data)
    workspace.face_data = data
    assembly = workspace.assembly
    revision = interface.phase_aperture_updates[]
    !topology_changed && assembly.revision == revision && return assembly
    flags = _mimetic_face_sides!(assembly.flags, interface, data.liquid_aperture)
    gridline = _mimetic_gridline_faces!(workspace.gridline, interface)
    boundary_aperture = resize!(workspace.boundary_aperture, length(mesh.boundary_faces))
    boundary_low = resize!(workspace.boundary_low, length(mesh.boundary_faces))
    for (index, face) in pairs(mesh.boundary_faces)
        boundary_aperture[index] = _phase_boundary_face_apertures(interface, nothing, index)[1]
        fraction = _conditional_fraction(interface, face.cell)
        boundary_low[index] = zero(T) < fraction < one(T) && first(
            _mimetic_trace_vote(
                interface, face.cell, face.axis, face.side > 0 ? 0.5 : -0.5,
            ),
        ) > 0
    end
    _mimetic_assemble_from_data!(
        workspace, mesh, data, flags, boundary_aperture,
        boundary_low, solver.density; allow_wall_cuts = solver.allow_wall_cuts,
        gridline,
    )
    assembly.revision = revision
    return assembly
end

"""Assemble `𝓜`, factorize its band block and build the reduced divergence
operators from given face geometry (see `_assemble_cut_cell_metric!`)."""
function _mimetic_assemble_from_data!(
        workspace::MimeticProjectionWorkspace{T}, mesh, data, flags,
        boundary_aperture, boundary_low, density; allow_wall_cuts = false,
        gridline = nothing,
    ) where {T}
    if !_mimetic_topology_current(workspace.topology, mesh)
        workspace.topology = _mimetic_topology(mesh)
        workspace.references = _cell_references(mesh)
    end
    topology = workspace.topology
    assembly = _assemble_cut_cell_metric!(
        workspace.assembly, mesh, topology, data, flags,
        boundary_aperture, boundary_low, density; allow_wall_cuts, gridline,
    )
    workspace.factor = isempty(assembly.band) ? nothing : lu(assembly.band_matrix)
    cells = size(topology.sides, 2)
    # Mixture divergence columns: +|f| a on the minus cell, -|f| a on the plus cell.
    band_count = 2length(assembly.band)
    diagonal_count = 4(length(assembly.dof_face) - length(assembly.band))
    rows = sizehint!(Int[], band_count)
    columns = sizehint!(Int[], band_count)
    values = sizehint!(T[], band_count)
    lrows = sizehint!(Int[], diagonal_count)
    lcolumns = sizehint!(Int[], diagonal_count)
    lvalues = sizehint!(T[], diagonal_count)
    for dof in eachindex(assembly.dof_face)
        aperture = assembly.dof_aperture[dof]
        iszero(aperture) && continue
        face = assembly.dof_face[dof]
        weight = mesh.faces[face].area * aperture
        minus = topology.face_cells[1, face]
        plus = topology.face_cells[2, face]
        position = assembly.band_position[dof]
        if position > 0
            push!(rows, minus, plus)
            push!(columns, position, position)
            push!(values, weight, -weight)
        else
            stiffness = weight^2 / assembly.diagonal[dof]
            push!(lrows, minus, plus, minus, plus)
            push!(lcolumns, minus, plus, plus, minus)
            push!(lvalues, stiffness, stiffness, -stiffness, -stiffness)
        end
    end
    workspace.divergence_band = sparse(rows, columns, values, cells, length(assembly.band))
    workspace.divergence_band_transpose = sparse(workspace.divergence_band')
    workspace.diagonal_laplacian = sparse(lrows, lcolumns, lvalues, cells, cells)
    resize!(workspace.boundary_flux, cells)
    fill!(workspace.boundary_flux, zero(T))
    count = length(assembly.band)
    resize!(workspace.band_scratch, count)
    resize!(workspace.band_solution, count)
    resize!(workspace.cell_residual, cells)
    resize!(workspace.cell_scratch, cells)
    resize!(workspace.multiplier, cells)
    # Face coefficient of the preconditioning pressure operator `D W⁻¹ Dᵀ`, in
    # units of 1/rho: Σ_π (a|f|)² / W_r with W_r the diagonal phase mass.
    beta = resize!(workspace.preconditioner_coefficient, length(mesh.faces))
    fill!(beta, zero(T))
    for dof in eachindex(assembly.dof_face)
        aperture = assembly.dof_aperture[dof]
        iszero(aperture) && continue
        face = assembly.dof_face[dof]
        beta[face] += aperture^2 * mesh.faces[face].area^2 / assembly.dof_mass[dof]
    end
    return assembly
end

function _mimetic_boundary_flux!(flux, solver, workspace, velocity)
    interface = solver.interface
    topology = workspace.topology
    fill!(flux, zero(eltype(flux)))
    nx, ny = interface.mesh.block_size
    for (index, face) in pairs(interface.mesh.boundary_faces)
        liquid, gas = _phase_boundary_face_apertures(interface, nothing, index)
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
    if !isnothing(workspace.factor)
        band = workspace.band_scratch
        mul!(band, workspace.divergence_band_transpose, x)
        ldiv!(workspace.band_solution, workspace.factor, band)
        mul!(y, workspace.divergence_band, workspace.band_solution, true, true)
    end
    return y
end

function mul!(destination, op::_MimeticSchurOperator{T}, source) where {T}
    work = op.workspace
    full = work.cell_scratch
    full[1] = zero(T)
    copyto!(full, 2, source, 1, op.size)
    result = work.cell_residual
    _mimetic_schur_full!(result, work, full)
    copyto!(destination, 1, result, 2, op.size)
    return destination
end

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
liquid minus cell, `-1`: liquid plus cell). Cell fractions are snapped to
exactly zero and one below the interface's fraction floor, so the test is
exact."""
function _mimetic_gridline_faces!(gridline, interface)
    mesh = interface.mesh
    resize!(gridline, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        minus = _conditional_fraction(interface, face.minus)
        plus = _conditional_fraction(interface, face.plus)
        gridline[index] = minus == 1 && iszero(plus) ? Int8(1) :
            iszero(minus) && plus == 1 ? Int8(-1) : Int8(0)
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
    ldiv!(solution, workspace.factor, scratch)
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
        ldiv!(workspace.band_solution, workspace.factor, workspace.band_scratch)
        for (position, dof) in pairs(band)
            velocity[dof] += workspace.band_solution[position]
        end
    end
    for dof in eachindex(velocity)
        assembly.band_position[dof] > 0 && continue
        aperture = assembly.dof_aperture[dof]
        iszero(aperture) && continue
        face = assembly.dof_face[dof]
        weight = topology.spacing * aperture
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
        flux = topology.spacing * aperture * values[dof]
        outflow[topology.face_cells[1, face]] += flux
        outflow[topology.face_cells[2, face]] -= flux
    end
    return outflow
end

"""Gauge-reduced V-cycle of the pressure operator with the face coefficient
`workspace.preconditioner_coefficient` (by default `D W⁻¹ Dᵀ`, the pressure
operator of the diagonal phase-mass metric)."""
function _mimetic_preconditioner!(solver, workspace)
    interface = solver.interface
    cells = size(workspace.topology.sides, 2)
    operator = workspace.pressure_operator
    fresh = !(operator isa EllipticOperator) || operator.mesh !== interface.mesh ||
        length(operator.volumes) != cells
    if fresh
        operator = PoissonOperator(interface; density = solver.density)
        workspace.pressure_operator = operator
    end
    _replace_face_coefficient(operator, workspace.preconditioner_coefficient)
    multigrid = workspace.multigrid
    if !fresh && multigrid isa TreeGeometricMultigridPreconditioner
        _refresh_geometric_multigrid!(multigrid, operator)
    else
        multigrid = TreeGeometricMultigridPreconditioner(operator)
        workspace.multigrid = multigrid
    end
    previous = workspace.preconditioner
    workspace.preconditioner = _cmdgfm_reduced_pressure(
        multigrid, operator, collect(2:cells), ones(Int, cells - 1), nothing;
        reuse = previous isa CMDGFMReducedPressurePreconditioner ? previous : nothing,
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
value in the first cell is kept. Forces must therefore not subtract a
pressure gradient from the predictor; any `𝓜⁻¹Dᵀ` gradient in the predictor
is removed exactly.

The saddle-point system for the band face values and the pressure is solved
with the conjugate-gradient method on its pressure Schur complement, which is
applied matrix-free with a sparse factorization of the band block of `𝓜`;
one geometric-multigrid V-cycle of the pressure operator `D W⁻¹ Dᵀ` of the
diagonal phase-mass metric preconditions it. `_inherited_cell_flux` and
`_reference_dt` have the meaning of the CMDGFM projection.
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
    # Warm start from the previous pressure; the first cell is the anchor.
    cells = length(workspace.references)
    multiplier = workspace.multiplier
    anchor_pressure = _get(pressure, workspace.references[1])
    for cell in 1:cells
        multiplier[cell] = T(dt) * (_get(pressure, workspace.references[cell]) - anchor_pressure)
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
        _set!(pressure, anchor_pressure + multiplier[cell] / T(dt), workspace.references[cell])
    end
    fill_halo!(pressure)
    return MimeticProjectionStats(
        result.target_defect <= T(tolerance), result.iterations,
        result.initial_residual, result.final_residual, result.mixture_divergence,
        T(dt) * result.mixture_divergence, result.target_defect,
        result.energy_identity_defect, result.momentum_change,
        result.momentum_wall_force, result.transported_momentum_change,
        result.transported_momentum, cells - 1, length(assembly.band),
        length(projected), assembly.cut_cells, assembly.slaved_values,
        assembly.stabilised_values, assembly.lumped_values, assembly.defective_cells,
        assembly.gridline_faces,
        :pressure_multigrid,
    )
end

"""Solve the projection for the gathered predictor `workspace.predictor`.

`workspace.multiplier` holds the initial guess of `λ = dt p` (zero in the
anchor cell) and returns the solution; `workspace.velocity` returns the
projected phase values. The target outflow is `-weight` times the inherited
(negative integrated) cell flux. `removed_energy` is the jump energy removed
by gathering (see `_mimetic_gather!`); it enters the energy identity.
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
    volume = topology.spacing^2
    defect_of(outflow) = dt * maximum(c -> abs(outflow[c] - target(c)), 1:cells) / volume
    multiplier = workspace.multiplier
    multiplier[1] = zero(T)
    initial_residual = norm(_mimetic_outflow!(workspace.cell_scratch, workspace, predictor))
    _mimetic_recover!(workspace, multiplier)
    outflow = _mimetic_outflow!(workspace.cell_residual, workspace, workspace.velocity)
    reduced = cells - 1
    resize!(workspace.reduced_rhs, reduced)
    resize!(workspace.reduced_solution, reduced)
    if isnothing(workspace.krylov) || length(workspace.krylov.x) != reduced
        workspace.krylov = Krylov.CgWorkspace(reduced, reduced, Vector{T})
    end
    schur = _MimeticSchurOperator{T, typeof(workspace)}(workspace, reduced)
    iterations = 0
    defect = defect_of(outflow)
    while defect > tolerance && iterations < max_iterations && reduced > 0
        # S δλ = target - outflow on the non-anchor cells.
        for cell in 2:cells
            workspace.reduced_rhs[cell - 1] = target(cell) - outflow[cell]
        end
        rhs_norm = norm(workspace.reduced_rhs)
        (isfinite(rhs_norm) && rhs_norm > zero(T)) || break
        requested = min(tolerance, tolerance / max(defect, eps(T)))
        Krylov.cg!(
            workspace.krylov, schur, workspace.reduced_rhs;
            M = preconditioner, ldiv = true, atol = zero(T), rtol = requested,
            itmax = max_iterations - iterations,
        )
        correction = Krylov.solution(workspace.krylov)
        attempt = Krylov.iteration_count(workspace.krylov)
        iterations += attempt
        (all(isfinite, correction) && attempt > 0) || break
        for cell in 2:cells
            multiplier[cell] += correction[cell - 1]
        end
        _mimetic_recover!(workspace, multiplier)
        outflow = _mimetic_outflow!(workspace.cell_residual, workspace, workspace.velocity)
        previous = defect
        defect = defect_of(outflow)
        defect < previous || break
    end
    final_residual = norm(outflow)
    mixture_divergence = maximum(abs, outflow) / volume

    # Diagnostics in the metric 𝓜 and the transported momentum.
    projected = workspace.velocity
    mp = _mimetic_metric_product!(workspace.metric_predictor, workspace, predictor)
    mv = _mimetic_metric_product!(workspace.metric_velocity, workspace, projected)
    # The removed grid-line jump energy is 𝓜-orthogonal to the identified
    # space: it adds to E(u*) and to E(u* - u⁺) alike.
    energy_before = dot(predictor, mp) + removed_energy
    energy_after = dot(projected, mv)
    energy_correction = energy_before - 2dot(projected, mp) + energy_after
    identity = energy_before > zero(T) ?
        (energy_before - energy_after - energy_correction) / energy_before : zero(T)
    momentum = zeros(T, 2)
    wall = zeros(T, 2)
    transported = zeros(T, 2)
    transported_before = zeros(T, 2)
    for dof in eachindex(predictor)
        face = assembly.dof_face[dof]
        axis = topology.sides[2, topology.face_cells[1, face]] == face ? 1 : 2
        momentum[axis] += mv[dof] - mp[dof]
        mass = assembly.dof_mass[dof]
        transported[axis] += mass * (projected[dof] - predictor[dof])
        transported_before[axis] += mass * predictor[dof]
        wall[axis] += topology.spacing * assembly.dof_aperture[dof] * (
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
`[C > ½]` (CGFM) or the liquid fraction (CSF)."""
@inline _mimetic_indicator(formulation, fraction) = formulation isa CSF ? fraction :
    fraction > 0.5 ? one(fraction) : zero(fraction)

"""Cell curvature `κ_c` of the per-phase capillary potentials, in
`workspace.cell_curvature` (`NaN` where none is available).

A cut cell keeps its own (height-function) curvature. Every other cell, and a
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
        for k in 1:4
            face = topology.sides[k, cell]
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
    debris = T(_MIMETIC_DEBRIS_CURVATURE[])
    debris > zero(T) || return kappa
    # Study switch: debris cut cells (min(C, 1 - C) < ε or no valid own
    # curvature) and every cell whose curvature came only from debris get κ = 0.
    zeroed = falses(cells)
    for cell in 1:cells
        fraction = _conditional_fraction(interface, references[cell])
        zero(T) < fraction < one(T) || continue
        value = _get(interface.curvature, references[cell])
        (min(fraction, one(T) - fraction) < debris || !isfinite(value)) && (zeroed[cell] = true)
    end
    for cell in 1:cells
        zeroed[cell] && (kappa[cell] = zero(T); weight[cell] = zero(T))
    end
    for cell in 1:cells
        fraction = _conditional_fraction(interface, references[cell])
        (zeroed[cell] || zero(T) < fraction < one(T)) && continue
        total = weights = zero(T)
        near_debris = false
        for k in 1:4
            face = topology.sides[k, cell]
            face > 0 || continue
            minus = topology.face_cells[1, face]
            neighbour = minus == cell ? topology.face_cells[2, face] : minus
            near_debris |= zeroed[neighbour]
            w = weight[neighbour]
            w > zero(T) && (total += w * kappa[neighbour]; weights += w)
        end
        kappa[cell] = weights > zero(T) ? total / weights : near_debris ? zero(T) : kappa[cell]
    end
    return kappa
end

"""Study switch (study/impact-velocities; `0` = off, the default): with
`ε = _MIMETIC_DEBRIS_CURVATURE[] > 0`, cut cells with `min(C, 1 - C) < ε` or
without a finite own curvature carry no capillary potential (`κ = 0`), and
uncut cells take the weighted mean over the remaining cut neighbours only."""
const _MIMETIC_DEBRIS_CURVATURE = Ref(0.0)

"""Study switch (study/impact-velocities): `:perphase` (default) is the
per-phase force below; `:jump` gives every phase value the package face
jump `a^π σ κ_f [I]_f |f|` (CGFM/CSF face curvature), as the identified
grid-line values, so cells without an indicator jump (gas debris inside the
liquid) exert no force."""
const _MIMETIC_FORCE_MODE = Ref(:perphase)

"""Add the per-phase surface-tension increment `dt 𝓜⁻¹(a^π J^π)` to each phase.

Each phase has its own capillary potential, `φ^g = σκI` and
`φ^l = σκI - σκ` with the cell curvature `κ` of
[`_mimetic_cell_curvature!`](@ref) and the indicator `I` (Heaviside `[C > ½]`
for CGFM, the liquid fraction for CSF), and the value of phase `π` on face `f`
gets the integrated jump `J^π_f = |f| (φ^π_+ - φ^π_-)`. The two potentials
differ by `σκ`, the capillary pressure jump, so a static drop of constant
curvature is exactly balanced (both jumps are then the common CGFM/CSF jump
`σκ [I]_f |f|`, a gradient), while a varying curvature drives only the liquid
(up to a common gradient the force is `-σ a^l |f| Δ_f κ` on the liquid alone).
An identified grid-line value (one value for both phases on a face between a
pure liquid and a pure gas cell) gets the common jump `σ κ_f [I]_f |f|` with
the face curvature of [`surface_tension!`](@ref). A face whose force needs a
curvature that is not available gets no force; such faces are counted in
`unresolved_curvature_faces`. `curvature` prescribes a constant curvature
(tests of the balance)."""
function _mimetic_capillary_force!(
        velocity::TwoPhaseVelocity, solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, dt;
        sigma, formulation = CGFM(), curvature = nothing,
        unresolved_curvature_faces::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {T}
    _validate_surface_tension_formulation(formulation)
    isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
    isnothing(curvature) || isfinite(curvature) || throw(ArgumentError("constant curvature must be finite"))
    interface = solver.interface
    mesh = interface.mesh
    assembly = _mimetic_assemble!(solver, workspace)
    topology = workspace.topology
    count = length(assembly.band)
    resize!(workspace.band_scratch, count)
    resize!(workspace.band_solution, count)
    kappa = _mimetic_cell_curvature!(workspace, interface, curvature)
    rhs = workspace.force_rhs
    resize!(rhs, length(assembly.dof_face))
    fill!(rhs, zero(T))
    sigma = T(sigma)
    unresolved = 0
    for (face, entry) in pairs(mesh.faces)
        liquid, gas = assembly.liquid_dof[face], assembly.gas_dof[face]
        minus, plus = topology.face_cells[1, face], topology.face_cells[2, face]
        cm = _conditional_fraction(interface, entry.minus)
        cp = _conditional_fraction(interface, entry.plus)
        if (liquid > 0 && liquid == gas) || _MIMETIC_FORCE_MODE[] === :jump
            jump = _surface_tension_jump(formulation, cm, cp)
            (iszero(sigma) || iszero(jump)) && continue
            value = isnothing(curvature) ? _face_curvature(interface, entry) : curvature
            if isnothing(value)
                unresolved += 1
            else
                # Study switch `:jump`: every value gets a^π σ κ_f [I]_f |f|.
                jump_force = sigma * value * jump * entry.area
                if liquid == gas
                    rhs[liquid] = jump_force
                else
                    liquid > 0 && (rhs[liquid] = assembly.dof_aperture[liquid] * jump_force)
                    gas > 0 && (rhs[gas] = assembly.dof_aperture[gas] * jump_force)
                end
            end
            continue
        end
        im, ip = _mimetic_indicator(formulation, cm), _mimetic_indicator(formulation, cp)
        # Potentials φ^g = σκI and φ^l = -σκ(1 - I); a vanishing factor needs
        # no curvature.
        gas_aperture = gas > 0 ? assembly.dof_aperture[gas] : zero(T)
        liquid_aperture = liquid > 0 ? assembly.dof_aperture[liquid] : zero(T)
        gas_minus = iszero(im) ? zero(T) : sigma * kappa[minus] * im
        gas_plus = iszero(ip) ? zero(T) : sigma * kappa[plus] * ip
        liquid_minus = isone(im) ? zero(T) : -sigma * kappa[minus] * (one(T) - im)
        liquid_plus = isone(ip) ? zero(T) : -sigma * kappa[plus] * (one(T) - ip)
        gas_jump = entry.area * (gas_plus - gas_minus)
        liquid_jump = entry.area * (liquid_plus - liquid_minus)
        if (gas_aperture > zero(T) && !isfinite(gas_jump)) ||
                (liquid_aperture > zero(T) && !isfinite(liquid_jump))
            unresolved += 1
            continue
        end
        gas_aperture > zero(T) && (rhs[gas] = gas_aperture * gas_jump)
        liquid_aperture > zero(T) && (rhs[liquid] = liquid_aperture * liquid_jump)
    end
    isnothing(unresolved_curvature_faces) || (unresolved_curvature_faces[] = unresolved)
    acceleration = _mimetic_metric_solve!(workspace.force_solution, workspace, rhs)
    for dof in eachindex(acceleration)
        _mimetic_scatter!(velocity, assembly, dof, T(dt) * acceleration[dof], true)
    end
    return velocity
end

"""Sparse matrix of the assembled metric `𝓜` on all degrees of freedom."""
function _mimetic_metric_matrix(workspace::MimeticProjectionWorkspace{T}) where {T}
    assembly = workspace.assembly
    dofs = length(assembly.dof_face)
    band = assembly.band
    rows, columns, values = findnz(assembly.band_matrix)
    for index in eachindex(rows)
        rows[index] = band[rows[index]]
        columns[index] = band[columns[index]]
    end
    for dof in 1:dofs
        assembly.band_position[dof] > 0 && continue
        push!(rows, dof)
        push!(columns, dof)
        push!(values, assembly.diagonal[dof])
    end
    return sparse(rows, columns, values, dofs, dofs)
end

"""Implicit-midpoint viscous step of the phase values in the metric `𝓜`.

With the mixture map `R = B⁻¹PᵀW` (the staggered-mass mean `ū = Ru` of the
values of a face, `B = PᵀWP`) and the symmetric positive semi-definite stress
matrix `K` of the mixture solve, `A = RᵀKR` and

    (𝓜 + dt/2 A) u⁺ = (𝓜 - dt/2 A) u.

The amplification factors `(1 - dtλ/2)/(1 + dtλ/2)` of the pencil `(A, 𝓜)` lie
in `(-1, 1]` for every `dt`: the step does not increase `uᵀ𝓜u`, the energy of
the projection, for any time step. For `𝓜 = W` it is the common increment of
the Crank--Nicolson mixture solve, `u⁺ = u + Pδ`. The values of `source` are
gathered as for the projection (an identified grid-line value is the
mass-weighted mean of the two phase values) and the result is written to
`destination`; absent (`NaN`) values stay absent. The system is solved
directly (sparse Cholesky); returns its relative residual and size."""
function _mimetic_viscous_step!(
        destination::TwoPhaseVelocity, source::TwoPhaseVelocity,
        solver::MimeticProjectionSolver{T}, workspace::MimeticProjectionWorkspace{T},
        stiffness::AbstractMatrix{T}, dt,
    ) where {T}
    assembly = _mimetic_assemble!(solver, workspace)
    dofs = length(assembly.dof_face)
    faces = size(stiffness, 1)
    values = workspace.viscous_values
    _mimetic_gather!(values, assembly, source)
    mass = resize!(workspace.viscous_face_mass, faces)
    fill!(mass, zero(T))
    for dof in 1:dofs
        mass[assembly.dof_face[dof]] += assembly.dof_mass[dof]
    end
    weights = [assembly.dof_mass[dof] / mass[assembly.dof_face[dof]] for dof in 1:dofs]
    mixture = sparse(assembly.dof_face, 1:dofs, weights, faces, dofs)
    viscous = mixture' * (stiffness * mixture)
    metric = _mimetic_metric_matrix(workspace)
    half = T(dt) / 2
    matrix = metric + half * viscous
    rhs = metric * values - half * (viscous * values)
    solution = cholesky(Symmetric(matrix)) \ rhs
    residual = norm(matrix * solution - rhs, Inf) / max(norm(rhs, Inf), floatmin(T))
    _copy_velocity!(destination, source)
    for dof in 1:dofs
        _mimetic_scatter!(destination, assembly, dof, solution[dof], false)
    end
    return (; residual, unknowns = dofs)
end

"""Add `dt g` to every active interior phase value: gravity as the uniform
acceleration of both phases. The body force `ρ_π g` is `𝓜` times the constant
field `g`, which the metric reproduces exactly; a density-weighted jump force
`𝓜⁻¹(a^π J)` would instead give the light phase value of a mixed face the
weight of the mixture."""
function _mimetic_uniform_acceleration!(velocity::TwoPhaseVelocity, mesh, gravity, dt)
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
    return velocity
end
