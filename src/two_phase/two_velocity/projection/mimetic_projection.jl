"""Configuration shared by all energy-preserving mimetic projections.

Only the jump model varies: `:divergence` selects the original centred
aperture-divergence condition, while `:face` and `:cell` select the unscaled
face- and cell-local aperture-normal conditions. `:commuting` constructs a
pressure-compatible tangential observation first and then derives the normal
constraint adjoint from it. The transpose of the selected constraint appears
in the jump stationarity equation, so every variant is an orthogonal
projection in the phase kinetic-energy inner product.
"""
struct MimeticProjectionSolver{T, I, P, PO, L, W}
    interface::I
    density::NTuple{2, T}
    preconditioner::P
    preconditioner_options::PO
    active_mass_fraction::T
    rank_tolerance::T
    jump_interpolant::L
    interpolation_weights::W
    commuting_patch_radius::Int
    commuting_max_patch_radius::Int
    commuting_amplification_limit::T
    commuting_accuracy_factor::T
    commuting_sliver_epsilon::T
end

function MimeticProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
        preconditioner = :cholesky, preconditioner_options = (;),
        active_mass_fraction = nothing, rank_tolerance = nothing,
        jump_interpolant = :divergence,
        interpolation_weights = nothing,
        commuting_patch_radius = 2, commuting_max_patch_radius = nothing,
        commuting_amplification_limit = 50,
        commuting_accuracy_factor = 1,
        commuting_sliver_epsilon = 0.05,
    ) where {T}
    rho = (T(density[1]), T(density[2]))
    all(value -> isfinite(value) && value > zero(T), rho) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    mass_floor = isnothing(active_mass_fraction) ?
        sqrt(interface.fraction_floor) : T(active_mass_fraction)
    isfinite(mass_floor) && zero(T) < mass_floor < T(0.5) || throw(
        ArgumentError("active_mass_fraction must be finite and in (0, 0.5)"),
    )
    row_tolerance = isnothing(rank_tolerance) ?
        sqrt(eps(T)) : T(rank_tolerance)
    isfinite(row_tolerance) && zero(T) < row_tolerance < one(T) || throw(
        ArgumentError("rank_tolerance must be finite and in (0, 1)"),
    )
    jump_interpolant in (:divergence, :cell, :face, :commuting) || throw(
        ArgumentError(
            "jump_interpolant must be :divergence, :cell, :face, or " *
                ":commuting",
        ),
    )
    weights = isnothing(interpolation_weights) ?
        (jump_interpolant === :commuting ? :geometric : :c1mc) :
        interpolation_weights
    weights in (:c1mc, :geometric) || throw(
        ArgumentError(
            "interpolation_weights must be :c1mc or :geometric",
        ),
    )
    jump_interpolant === :commuting && weights !== :geometric && throw(
        ArgumentError(
            "the commuting interpolant is defined by aperture geometry " *
                "and does not use c(1-c) weights",
        ),
    )
    patch_radius = Int(commuting_patch_radius)
    max_patch_radius = isnothing(commuting_max_patch_radius) ?
        typemax(Int) : Int(commuting_max_patch_radius)
    patch_radius >= 0 || throw(
        ArgumentError("commuting_patch_radius must be nonnegative"),
    )
    max_patch_radius >= patch_radius || throw(
        ArgumentError(
            "commuting_max_patch_radius must not be smaller than " *
                "commuting_patch_radius",
        ),
    )
    amplification_limit = T(commuting_amplification_limit)
    isfinite(amplification_limit) && amplification_limit > zero(T) || throw(
        ArgumentError(
            "commuting_amplification_limit must be finite and positive",
        ),
    )
    accuracy_factor = T(commuting_accuracy_factor)
    isfinite(accuracy_factor) && accuracy_factor > zero(T) || throw(
        ArgumentError(
            "commuting_accuracy_factor must be finite and positive",
        ),
    )
    sliver_epsilon = T(commuting_sliver_epsilon)
    isfinite(sliver_epsilon) &&
        zero(T) <= sliver_epsilon < T(0.5) || throw(
        ArgumentError(
            "commuting_sliver_epsilon must be finite and in [0, 0.5)",
        ),
    )
    return MimeticProjectionSolver{
        T, typeof(interface), typeof(preconditioner),
        typeof(preconditioner_options), typeof(jump_interpolant),
        typeof(weights),
    }(
        interface, rho, preconditioner, preconditioner_options,
        mass_floor, row_tolerance, jump_interpolant,
        weights, patch_radius, max_patch_radius,
        amplification_limit, accuracy_factor, sliver_epsilon,
    )
end

function _local_block_preconditioner(
        pressure_block, relative_diagonal, constraint, ::Type{T},
    ) where {T}
    np = size(pressure_block, 1)
    nx = length(relative_diagonal)
    nl = size(constraint, 1)
    zero_px = spzeros(T, np, nx)
    zero_pl = spzeros(T, np, nl)
    zero_xl = spzeros(T, nx, nl)
    jump_block = spdiagm(0 => relative_diagonal)
    multiplier_block = if nl == 0
        spzeros(T, 0, 0)
    else
        schur = constraint * spdiagm(0 => inv.(relative_diagonal)) *
            transpose(constraint)
        # Algebraically independent local rows can still be almost parallel.
        # A roundoff-sized shift belongs only to the preconditioner; the KKT
        # operator and its exact constraints are left unchanged.
        scale = maximum(abs, diag(schur); init = one(T))
        shift = T(256) * eps(T) * scale
        schur + spdiagm(0 => fill(shift, nl))
    end
    return [
        pressure_block zero_px zero_pl
        transpose(zero_px) jump_block zero_xl
        transpose(zero_pl) transpose(zero_xl) multiplier_block
    ]
end

"""Reusable fields and the most recently assembled local KKT system."""
mutable struct MimeticProjectionWorkspace{T, M <: BlockTreeMesh}
    pressure_gradient::FaceVelocity{T, M}
    base_gradient::FaceVelocity{T, M}
    liquid_gradient::FaceVelocity{T, M}
    gas_gradient::FaceVelocity{T, M}
    jump::Vector{T}
    liquid_fraction_x::FaceField{1, T, M}
    liquid_fraction_y::FaceField{2, T, M}
    matrix::SparseMatrixCSC{T, Int}
    block_preconditioner::SparseMatrixCSC{T, Int}
    constraint_matrix::SparseMatrixCSC{T, Int}
    right_hand_side::Vector{T}
    solution::Vector{T}
    pressure_indices::Vector{Int}
    active_faces::Vector{Int}
    mixed_faces::Vector{Int}
    constraint_locations::Vector{Int}
    constraint_scales::Vector{T}
end

function MimeticProjectionWorkspace(
        solver::MimeticProjectionSolver{T},
    ) where {T}
    mesh = solver.interface.mesh
    return MimeticProjectionWorkspace(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        zeros(T, length(mesh.faces)), FaceField{1}(mesh, T),
        FaceField{2}(mesh, T), spzeros(T, 0, 0), spzeros(T, 0, 0),
        spzeros(T, 0, 0), T[], T[], Int[], Int[], Int[], Int[], T[],
    )
end

"""Diagnostics for one local, energy-compatible projection."""
struct MimeticProjectionStats{T}
    converged::Bool
    iterations::Int
    initial_residual::T
    final_residual::T
    matrix_symmetry_error::T
    mixture_divergence::T
    mixture_volume_defect::T
    normal_jump_divergence::T
    normal_volume_defect::T
    normal_velocity_defect::T
    tangential_defect::T
    target_defect::T
    momentum_error::NTuple{2, T}
    momentum_identity_error::T
    energy_identity_error::T
    orthogonality_error::T
    pressure_unknowns::Int
    jump_unknowns::Int
    multiplier_unknowns::Int
    candidate_constraint_rows::Int
    independent_constraint_rows::Int
    dependent_constraint_rows::Int
    underresolved_constraint_rows::Int
    commuting_constraint_rows::Int
    exact_commuting_constraint_rows::Int
    approximate_commuting_constraint_rows::Int
    fallback_constraint_rows::Int
    commuting_residual::T
    pressure_commuting_constraint_rows::Int
    pressure_fallback_constraint_rows::Int
    pressure_commuting_residual::T
    mass_deactivated_faces::Int
    embedded_deactivated_faces::Int
    sliver_extrapolated_faces::Int
    sliver_zeroed_faces::Int
    commuting_sliver_epsilon::T
    jump_interpolant::Symbol
    interpolation_weights::Symbol
    preconditioner::Symbol
end

_dimensionless_residual(stats::MimeticProjectionStats) = max(
    stats.mixture_volume_defect,
    stats.normal_volume_defect,
    stats.tangential_defect,
)

function Base.show(io::IO, stats::MimeticProjectionStats)
    return print(
        io, "MimeticProjectionStats(converged=", stats.converged,
        ", iterations=", stats.iterations,
        ", initial_residual=", stats.initial_residual,
        ", final_residual=", stats.final_residual,
        ", matrix_symmetry_error=", stats.matrix_symmetry_error,
        ", mixture_divergence=", stats.mixture_divergence,
        ", mixture_volume_defect=", stats.mixture_volume_defect,
        ", normal_jump_divergence=", stats.normal_jump_divergence,
        ", normal_volume_defect=", stats.normal_volume_defect,
        ", normal_velocity_defect=", stats.normal_velocity_defect,
        ", tangential_defect=", stats.tangential_defect,
        ", target_defect=", stats.target_defect,
        ", momentum_error=", stats.momentum_error,
        ", momentum_identity_error=", stats.momentum_identity_error,
        ", energy_identity_error=", stats.energy_identity_error,
        ", orthogonality_error=", stats.orthogonality_error,
        ", unknowns=(`p`=", stats.pressure_unknowns,
        ", `xi`=", stats.jump_unknowns,
        ", `lambda`=", stats.independent_constraint_rows, ")",
        ", constraint_rows=(candidate=", stats.candidate_constraint_rows,
        ", dependent=", stats.dependent_constraint_rows,
        ", underresolved=", stats.underresolved_constraint_rows,
        ", commuting=", stats.commuting_constraint_rows,
        ", exact_commuting=", stats.exact_commuting_constraint_rows,
        ", approximate_commuting=",
        stats.approximate_commuting_constraint_rows,
        ", fallback=", stats.fallback_constraint_rows, ")",
        ", commuting_residual=", stats.commuting_residual,
        ", pressure_commuting_rows=",
        stats.pressure_commuting_constraint_rows,
        ", pressure_fallback_rows=",
        stats.pressure_fallback_constraint_rows,
        ", pressure_commuting_residual=",
        stats.pressure_commuting_residual,
        ", deactivated=(mass=", stats.mass_deactivated_faces,
        ", embedded=", stats.embedded_deactivated_faces, ")",
        ", slivers=(epsilon=", stats.commuting_sliver_epsilon,
        ", extrapolated=", stats.sliver_extrapolated_faces,
        ", zeroed=", stats.sliver_zeroed_faces, ")",
        ", jump_interpolant=", stats.jump_interpolant,
        ", interpolation_weights=", stats.interpolation_weights,
        ", preconditioner=", stats.preconditioner, ")",
    )
end

function _local_embedded_cells(interface::InterfaceField{T}) where {T}
    geometry = interface.geometry
    result = falses(ncells(interface.mesh))
    isnothing(geometry) && return result
    references = _cell_references(interface.mesh)
    threshold = interface.fraction_floor
    for (index, reference) in pairs(references)
        open_fraction = _get(geometry.open_fraction, reference)
        @inbounds result[index] = open_fraction < one(T) - threshold
    end
    return result
end

function _local_active_faces(solver, data)
    interface = solver.interface
    mesh = interface.mesh
    nx, ny = mesh.block_size
    embedded_cells = _local_embedded_cells(interface)
    active = Int[]
    mass_deactivated = 0
    embedded_deactivated = 0
    for face_index in data.mixed
        face = mesh.faces[face_index]
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        if embedded_cells[minus] || embedded_cells[plus]
            embedded_deactivated += 1
            continue
        end
        total = @inbounds data.total_mass[face_index]
        relative_fraction = @inbounds min(
            data.liquid_mass[face_index], data.gas_mass[face_index],
        ) / total
        open_aperture = @inbounds data.liquid_aperture[face_index] +
            data.gas_aperture[face_index]
        if relative_fraction < solver.active_mass_fraction ||
                open_aperture <= interface.fraction_floor
            mass_deactivated += 1
            continue
        end
        push!(active, face_index)
    end
    return active, mass_deactivated, embedded_deactivated
end

"""Reduce the commuting jump space by constant sliver extrapolation.

Let `xi = E * xi_hat`, where rows of `E` are the original active mixed
faces and columns are non-sliver jump unknowns. A non-sliver row selects its
own unknown. A sliver row selects one of the two parallel faces which meet
its tangential endpoints: among eligible non-sliver neighbours, the donor
whose staggered liquid fraction is closest to `1/2` is used. If neither
endpoint has an eligible donor, that row of `E` is zero.

The reduction is applied by congruence to the jump Riesz map and by the same
right action to every pressure/constraint coupling. Consequently it is part
of the variational space, rather than a post-processing overwrite, and the
symmetric momentum- and energy-compatible KKT structure is unchanged.
"""
function _local_commuting_jump_space(
        solver::MimeticProjectionSolver{T}, data, active_faces,
        relative_diagonal, face_dual_volumes,
    ) where {T}
    face_count = length(active_faces)
    epsilon = solver.commuting_sliver_epsilon
    if solver.jump_interpolant !== :commuting || iszero(epsilon)
        extension = sparse(
            1:face_count, 1:face_count, ones(T, face_count),
            face_count, face_count,
        )
        residual_scales = [
            data.total_mass[face] * face_dual_volumes[local_face] *
                sqrt(face_dual_volumes[local_face])
            for (local_face, face) in pairs(active_faces)
        ]
        return (;
            full_active_faces = active_faces,
            active_faces,
            extension,
            relative_diagonal,
            face_dual_volumes,
            residual_scales,
            sliver_faces = Int[],
            sliver_donor_faces = Int[],
            extrapolated_count = 0,
            zeroed_count = 0,
            identity = true,
        )
    end

    mesh = solver.interface.mesh
    active_lookup = zeros(Int, length(mesh.faces))
    for (local_face, global_face) in pairs(active_faces)
        active_lookup[global_face] = local_face
    end
    valid = BitVector(undef, face_count)
    for (local_face, global_face) in pairs(active_faces)
        fraction = data.liquid_fraction[global_face]
        valid[local_face] = epsilon <= fraction <= one(T) - epsilon
    end
    active_to_reduced = zeros(Int, face_count)
    reduced_faces = Int[]
    for local_face in eachindex(active_faces)
        valid[local_face] || continue
        active_to_reduced[local_face] = length(reduced_faces) + 1
        push!(reduced_faces, active_faces[local_face])
    end

    # A same-axis face sharing either endpoint is the immediate parallel
    # neighbour reached in the tangential direction. On a uniform mesh this
    # gives exactly the two requested candidates; the vertex representation
    # also handles periodic seams and coarse--fine face segmentation.
    vertex_patches = _vertex_patches(mesh, T)
    extension_rows = Int[]
    extension_columns = Int[]
    extension_values = T[]
    sliver_faces = Int[]
    sliver_donor_faces = Int[]
    extrapolated_count = 0
    zeroed_count = 0
    for (local_face, global_face) in pairs(active_faces)
        if valid[local_face]
            push!(extension_rows, local_face)
            push!(extension_columns, active_to_reduced[local_face])
            push!(extension_values, one(T))
            continue
        end

        push!(sliver_faces, global_face)
        face = mesh.faces[global_face]
        candidates = Int[]
        for endpoint in 1:2
            patch = get(
                vertex_patches, _vertex_key(mesh, face, endpoint), nothing,
            )
            isnothing(patch) && continue
            for candidate in patch
                candidate.index == global_face && continue
                candidate.axis == face.axis || continue
                candidate_local = active_lookup[candidate.index]
                candidate_local > 0 || continue
                valid[candidate_local] || continue
                candidate_local in candidates ||
                    push!(candidates, candidate_local)
            end
        end
        if isempty(candidates)
            push!(sliver_donor_faces, 0)
            zeroed_count += 1
            continue
        end

        donor_local = first(candidates)
        donor_score = abs(
            data.liquid_fraction[active_faces[donor_local]] - T(0.5),
        )
        for candidate_local in Iterators.drop(candidates, 1)
            candidate_face = active_faces[candidate_local]
            candidate_score = abs(
                data.liquid_fraction[candidate_face] - T(0.5),
            )
            donor_face = active_faces[donor_local]
            if candidate_score < donor_score ||
                    (candidate_score == donor_score &&
                        candidate_face < donor_face)
                donor_local = candidate_local
                donor_score = candidate_score
            end
        end
        donor_face = active_faces[donor_local]
        push!(sliver_donor_faces, donor_face)
        push!(extension_rows, local_face)
        push!(extension_columns, active_to_reduced[donor_local])
        push!(extension_values, one(T))
        extrapolated_count += 1
    end
    reduced_count = length(reduced_faces)
    extension = sparse(
        extension_rows, extension_columns, extension_values,
        face_count, reduced_count,
    )

    # Since every retained row of E contains one unit entry, E' K_xi E is
    # diagonal: tied sliver masses are accumulated into their donor unknown.
    reduced_diagonal = zeros(T, reduced_count)
    reduced_dual_volumes = zeros(T, reduced_count)
    residual_scales = zeros(T, reduced_count)
    extension_row_values = rowvals(extension)
    extension_values_nz = nonzeros(extension)
    for column in axes(extension, 2)
        for pointer in nzrange(extension, column)
            local_face = extension_row_values[pointer]
            value = extension_values_nz[pointer]
            global_face = active_faces[local_face]
            dual_volume = face_dual_volumes[local_face]
            reduced_diagonal[column] +=
                value^2 * relative_diagonal[local_face]
            reduced_dual_volumes[column] += value^2 * dual_volume
            residual_scales[column] += abs(value) *
                data.total_mass[global_face] * dual_volume *
                sqrt(dual_volume)
        end
    end
    return (;
        full_active_faces = active_faces,
        active_faces = reduced_faces,
        extension,
        relative_diagonal = reduced_diagonal,
        face_dual_volumes = reduced_dual_volumes,
        residual_scales,
        sliver_faces,
        sliver_donor_faces,
        extrapolated_count,
        zeroed_count,
        identity = false,
    )
end

"""Cell-integrated `D(a_l n)` vectors, including physical-boundary faces."""
function _local_integrated_aperture_vectors(interface, data)
    T = eltype(interface)
    mesh = interface.mesh
    nx, ny = mesh.block_size
    x = zeros(T, ncells(mesh))
    y = zeros(T, ncells(mesh))
    for (index, face) in pairs(mesh.faces)
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        contribution = T(face.area) * data.liquid_aperture[index]
        component = face.axis == 1 ? x : y
        @inbounds begin
            component[minus] += contribution
            component[plus] -= contribution
        end
    end
    for (index, face) in pairs(mesh.boundary_faces)
        liquid_aperture, = _phase_boundary_face_apertures(interface, index)
        cell = _elliptic_cell_index(face.cell, nx, ny)
        component = face.axis == 1 ? x : y
        @inbounds component[cell] +=
            T(face.side) * T(face.area) * liquid_aperture
    end
    return x, y
end

function _local_independent_rows(
        matrix, relative_diagonal, tolerance, ::Type{T},
    ) where {T}
    size(matrix, 1) == 0 && return Int[]
    # Rank must be measured in the multiplier Schur metric, not merely in H.
    # This catches rows which are independent algebraically but indistinguish-
    # able after the phase kinetic Riesz map becomes strongly nonuniform.
    weighted = matrix * spdiagm(0 => inv.(sqrt.(relative_diagonal)))
    row_norms = sqrt.(diag(weighted * transpose(weighted)))
    normalised = spdiagm(0 => inv.(row_norms)) * weighted
    factorisation = qr(transpose(normalised); tol = tolerance)
    numerical_rank = rank(factorisation)
    numerical_rank == 0 && return Int[]
    return sort!(factorisation.pcol[1:numerical_rank])
end

@inline function _local_face_aperture_vector(
        mesh::BlockTreeMesh, face::InteriorFace, x, y,
    )
    nx, ny = mesh.block_size
    minus = _elliptic_cell_index(face.minus, nx, ny)
    plus = _elliptic_cell_index(face.plus, nx, ny)
    hm = mesh.blocks[face.minus.block].spacing[face.axis]
    hp = mesh.blocks[face.plus.block].spacing[face.axis]
    denominator = hm + hp
    return (
        (hm * x[minus] + hp * x[plus]) / denominator,
        (hm * y[minus] + hp * y[plus]) / denominator,
    )
end

"""Assemble the staggered `I D(a_l n) ⋅ J¹_f` constraint."""
function _local_face_constraint_operator(
        solver::MimeticProjectionSolver{T}, data, active_faces,
        relative_diagonal,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    active_lookup = zeros(Int, length(mesh.faces))
    for (local_index, face_index) in pairs(active_faces)
        active_lookup[face_index] = local_index
    end
    vertex_patches = _vertex_patches(mesh, T)
    aperture_x, aperture_y = _local_integrated_aperture_vectors(
        interface, data,
    )
    rows = Int[]
    columns = Int[]
    values = T[]
    candidate_faces = Int[]
    candidate_scales = T[]
    underresolved = 0
    coefficient_tolerance = T(64) * eps(T)
    for face_index in active_faces
        face = mesh.faces[face_index]
        q = _local_face_aperture_vector(
            mesh, face, aperture_x, aperture_y,
        )
        qnorm = hypot(q[1], q[2])
        if qnorm <= coefficient_tolerance * max(T(face.area), one(T))
            underresolved += 1
            continue
        end
        tangent = 3 - face.axis
        transverse = Int[]
        weights = T[]
        if abs(q[tangent]) > coefficient_tolerance * qnorm
            transverse = _cmdgfm_vertex_candidates(
                vertex_patches, mesh, face, face_index, tangent,
            )
            filter!(index -> active_lookup[index] > 0, transverse)
            raw_weights = T[]
            for index in transverse
                if solver.interpolation_weights === :c1mc
                    fraction = data.liquid_fraction[index]
                    push!(raw_weights, fraction * (one(T) - fraction))
                else
                    push!(raw_weights, T(mesh.faces[index].area))
                end
            end
            weight_sum = sum(raw_weights)
            touches_boundary = any(
                endpoint -> _physical_vertex(
                    mesh, _vertex_key(mesh, face, endpoint),
                ),
                1:2,
            )
            crosses_level = any(
                index -> mesh.faces[index].area != face.area, transverse,
            )
            required = touches_boundary || crosses_level ? 1 : 2
            if length(transverse) < required || weight_sum <= zero(T)
                underresolved += 1
                continue
            end
            weights = raw_weights ./ weight_sum
        end

        local_columns = Int[]
        local_values = T[]
        normal_coefficient = q[face.axis]
        if abs(normal_coefficient) > coefficient_tolerance * qnorm
            push!(local_columns, active_lookup[face_index])
            push!(local_values, normal_coefficient)
        end
        for (candidate, weight) in zip(transverse, weights)
            coefficient = q[tangent] * weight
            abs(coefficient) > coefficient_tolerance * qnorm || continue
            push!(local_columns, active_lookup[candidate])
            push!(local_values, coefficient)
        end
        if isempty(local_columns)
            underresolved += 1
            continue
        end
        row = length(candidate_faces) + 1
        append!(rows, fill(row, length(local_columns)))
        append!(columns, local_columns)
        append!(values, local_values)
        push!(candidate_faces, face_index)
        push!(candidate_scales, qnorm)
    end
    candidate = sparse(
        rows, columns, values, length(candidate_faces),
        length(active_faces),
    )
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    constraint = candidate[retained, :]
    return (;
        constraint, candidate_constraint = candidate,
        candidate_locations = candidate_faces,
        constraint_locations = candidate_faces[retained],
        constraint_scales = candidate_scales[retained],
        candidate_scales,
        candidate_count = length(candidate_faces),
        dependent_count = length(candidate_faces) - length(retained),
        underresolved,
    )
end

"""Assemble independent rows of `D(a_l n) ⋅ J¹_c` on the active jump space."""
function _local_cell_constraint_operator(
        solver::MimeticProjectionSolver{T}, data, active_faces,
        relative_diagonal,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    active_lookup = zeros(Int, length(mesh.faces))
    mixed_lookup = falses(length(mesh.faces))
    for (local_index, face_index) in pairs(active_faces)
        active_lookup[face_index] = local_index
    end
    mixed_lookup[data.mixed] .= true
    aperture_x, aperture_y = _local_integrated_aperture_vectors(
        interface, data,
    )
    incident_faces = _cmdgfm_incident_faces(mesh)
    references = _cell_references(mesh)
    embedded_cells = _local_embedded_cells(interface)
    rows = Int[]
    columns = Int[]
    values = T[]
    candidate_cells = Int[]
    candidate_scales = T[]
    underresolved = 0
    coefficient_tolerance = T(64) * eps(T)
    floor = interface.fraction_floor
    for (cell_index, reference) in pairs(references)
        fraction = _conditional_fraction(interface, reference)
        floor < fraction < one(T) - floor || continue
        embedded_cells[cell_index] && continue
        q = (aperture_x[cell_index], aperture_y[cell_index])
        qnorm = hypot(q[1], q[2])
        cell_scale = maximum(mesh.blocks[reference.block].spacing)
        if qnorm <= coefficient_tolerance * max(T(cell_scale), one(T))
            underresolved += 1
            continue
        end

        local_columns = Int[]
        local_values = T[]
        resolved = true
        for axis in 1:2
            abs(q[axis]) > coefficient_tolerance * qnorm || continue
            incident = filter(incident_faces[cell_index]) do face_index
                mesh.faces[face_index].axis == axis
            end
            if solver.interpolation_weights === :c1mc
                weighted_faces = filter(
                    face_index -> mixed_lookup[face_index], incident,
                )
                raw_weights = T[]
                for face_index in weighted_faces
                    fraction = data.liquid_fraction[face_index]
                    push!(
                        raw_weights,
                        T(mesh.faces[face_index].area) * fraction *
                            (one(T) - fraction),
                    )
                end
                weight_sum = sum(raw_weights)
                if weight_sum <= zero(T)
                    resolved = false
                    break
                end
                for (face_index, raw_weight) in zip(
                        weighted_faces, raw_weights,
                    )
                    column = active_lookup[face_index]
                    # Removed jump faces have zero final jump by construction.
                    column > 0 || continue
                    coefficient = q[axis] * raw_weight / weight_sum
                    abs(coefficient) > coefficient_tolerance * qnorm || continue
                    location = findfirst(==(column), local_columns)
                    if isnothing(location)
                        push!(local_columns, column)
                        push!(local_values, coefficient)
                    else
                        local_values[location] += coefficient
                    end
                end
            else
                side_measures = zeros(T, 2)
                for face_index in incident
                    face = mesh.faces[face_index]
                    side = face.minus == reference ? 2 : 1
                    side_measures[side] += T(face.area)
                end
                if any(iszero, side_measures)
                    # A physical-boundary interface cell needs the prescribed
                    # boundary jump in J¹_c. Until that affine term is
                    # assembled, classify this row as underresolved.
                    resolved = false
                    break
                end
                for face_index in incident
                    mixed_lookup[face_index] || continue
                    face = mesh.faces[face_index]
                    side = face.minus == reference ? 2 : 1
                    column = active_lookup[face_index]
                    # Pure and removed faces use the zero-jump extension, but
                    # remain in the side measure of this geometric average.
                    column > 0 || continue
                    coefficient = q[axis] * T(face.area) /
                        (T(2) * side_measures[side])
                    abs(coefficient) > coefficient_tolerance * qnorm || continue
                    location = findfirst(==(column), local_columns)
                    if isnothing(location)
                        push!(local_columns, column)
                        push!(local_values, coefficient)
                    else
                        local_values[location] += coefficient
                    end
                end
            end
        end
        if !resolved
            underresolved += 1
            continue
        end
        if isempty(local_columns)
            # Every interpolated jump in this row belongs to the eliminated
            # active-set complement and is already zero in the final state.
            continue
        end
        row = length(candidate_cells) + 1
        append!(rows, fill(row, length(local_columns)))
        append!(columns, local_columns)
        append!(values, local_values)
        push!(candidate_cells, cell_index)
        push!(candidate_scales, qnorm)
    end
    candidate = sparse(
        rows, columns, values, length(candidate_cells),
        length(active_faces),
    )
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    constraint = candidate[retained, :]
    return (;
        constraint, candidate_constraint = candidate,
        candidate_locations = candidate_cells,
        constraint_locations = candidate_cells[retained],
        constraint_scales = candidate_scales[retained],
        candidate_scales,
        candidate_count = length(candidate_cells),
        dependent_count = length(candidate_cells) - length(retained),
        underresolved,
    )
end

"""Build aperture-based normal and tangential jump measurements.

For a mixed cell ``c``, let ``eta_c = D(a_l n)_c`` be its integrated
aperture Gauss vector and let ``J¹_c`` reconstruct the two Cartesian jump
components from active staggered faces.  This routine assembles

    N_a j = eta_x J¹_x j + eta_y J¹_y j,
    T   j = -eta_y J¹_x j + eta_x J¹_y j.

Both observations retain the scale `|eta|`, which is the interface length for
a planar cut. `T` is used only to certify and construct commuting corrections;
it is not an additional equation in the pressure solve.

`J¹_c` is a purely geometric, constant-reproducing average of the active
face-normal jump samples of each Cartesian component. In particular this
construction does not use the legacy `c(1-c)` weights. Its only interface
geometry is the aperture Gauss vector `eta = D(a_l n)`; no PLIC normal or
interface endpoint enters the operator.
"""
function _local_aperture_normal_tangent_reconstruction(
        solver::MimeticProjectionSolver{T}, data, active_faces,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    active_lookup = zeros(Int, length(mesh.faces))
    for (local_index, face_index) in pairs(active_faces)
        active_lookup[face_index] = local_index
    end
    aperture_x, aperture_y = _local_integrated_aperture_vectors(
        interface, data,
    )
    incident_faces = _cmdgfm_incident_faces(mesh)
    references = _cell_references(mesh)
    embedded_cells = _local_embedded_cells(interface)
    normal_rows = Int[]
    normal_columns = Int[]
    normal_values = T[]
    tangent_rows = Int[]
    tangent_columns = Int[]
    tangent_values = T[]
    candidate_cells = Int[]
    candidate_scales = T[]
    tangent_resolved = Bool[]
    underresolved = 0
    coefficient_tolerance = T(64) * eps(T)
    floor = interface.fraction_floor

    for (cell_index, reference) in pairs(references)
        fraction = _conditional_fraction(interface, reference)
        floor < fraction < one(T) - floor || continue
        embedded_cells[cell_index] && continue
        eta = (aperture_x[cell_index], aperture_y[cell_index])
        measure = hypot(eta[1], eta[2])
        cell_scale = maximum(mesh.blocks[reference.block].spacing)
        if measure <= coefficient_tolerance * max(T(cell_scale), one(T))
            underresolved += 1
            continue
        end

        axis_columns = (Int[], Int[])
        axis_weights = (T[], T[])
        axis_resolved = trues(2)
        for axis in 1:2
            incident = filter(incident_faces[cell_index]) do face_index
                mesh.faces[face_index].axis == axis
            end
            # The jump space contains only active mixed faces. Normalising
            # their geometric measures makes J¹ reproduce a constant vector
            # exactly on that space, including one-sided/AMR configurations.
            # No volume-fraction-dependent c(1-c) factor is present.
            available = filter(
                face_index -> active_lookup[face_index] > 0, incident,
            )
            weight_sum = sum(
                face_index -> T(mesh.faces[face_index].area), available;
                init = zero(T),
            )
            if weight_sum <= zero(T)
                axis_resolved[axis] = false
                continue
            end
            for face_index in available
                push!(axis_columns[axis], active_lookup[face_index])
                push!(
                    axis_weights[axis],
                    T(mesh.faces[face_index].area) / weight_sum,
                )
            end
        end

        normal_needed = (
            abs(eta[1]) > coefficient_tolerance * measure,
            abs(eta[2]) > coefficient_tolerance * measure,
        )
        tangent_coefficients = (-eta[2], eta[1])
        tangent_needed = (
            abs(tangent_coefficients[1]) > coefficient_tolerance * measure,
            abs(tangent_coefficients[2]) > coefficient_tolerance * measure,
        )
        normal_ok = all(
            !normal_needed[axis] || axis_resolved[axis] for axis in 1:2
        )
        if !normal_ok
            underresolved += 1
            continue
        end

        new_row = length(candidate_cells) + 1
        normal_nonzero = false
        for axis in 1:2
            for (column, weight) in zip(
                    axis_columns[axis], axis_weights[axis],
                )
                normal_value = eta[axis] * weight
                if abs(normal_value) > coefficient_tolerance * measure
                    push!(normal_rows, new_row)
                    push!(normal_columns, column)
                    push!(normal_values, normal_value)
                    normal_nonzero = true
                end
                tangent_value = tangent_coefficients[axis] * weight
                if abs(tangent_value) > coefficient_tolerance * measure
                    push!(tangent_rows, new_row)
                    push!(tangent_columns, column)
                    push!(tangent_values, tangent_value)
                end
            end
        end
        # If every contributing jump belongs to the active-set complement,
        # the homogeneous zero extension already supplies this equation.
        normal_nonzero || continue
        push!(candidate_cells, cell_index)
        push!(candidate_scales, measure)
        push!(
            tangent_resolved,
            all(
                !tangent_needed[axis] || axis_resolved[axis]
                    for axis in 1:2
            ),
        )
    end

    row_count = length(candidate_cells)
    face_count = length(active_faces)
    return (;
        normal = sparse(
            normal_rows, normal_columns, normal_values,
            row_count, face_count,
        ),
        tangent = sparse(
            tangent_rows, tangent_columns, tangent_values,
            row_count, face_count,
        ),
        candidate_cells, candidate_scales, tangent_resolved,
        underresolved,
    )
end

function _local_sparse_row_supports(matrix)
    supports = [Int[] for _ in axes(matrix, 1)]
    matrix_rows = rowvals(matrix)
    for column in axes(matrix, 2)
        for pointer in nzrange(matrix, column)
            push!(supports[matrix_rows[pointer]], column)
        end
    end
    return supports
end

function _local_rows_at_columns(row_supports, number_of_columns)
    result = [Int[] for _ in 1:number_of_columns]
    for (row, support) in pairs(row_supports)
        for column in support
            push!(result[column], row)
        end
    end
    return result
end

function _local_patch_rows(seed, radius, row_supports, rows_at_column)
    selected = BitSet((seed,))
    frontier = Int[seed]
    for _ in 1:radius
        next_frontier = Int[]
        for row in frontier
            for column in row_supports[row]
                for neighbour in rows_at_column[column]
                    neighbour in selected && continue
                    push!(selected, neighbour)
                    push!(next_frontier, neighbour)
                end
            end
        end
        frontier = next_frontier
        isempty(frontier) && break
    end
    return collect(selected)
end

"""Kinetic minimum-change projection onto local linear constraints.

The objective metric is `inv(K_xi)`, hence its inverse in the projection
formula is the diagonal `K_xi`.  Constraint rows are first normalised in
that metric.  Rank-deficient but compatible rows are harmless; incompatibility,
poor nonzero conditioning, or excessive coefficient amplification causes a
certification failure and triggers patch growth or fallback.
"""
function _local_commuting_fit(
        baseline, constraints, target, kinetic;
        rank_tolerance, amplification_limit,
    )
    T = eltype(baseline)
    retained = Int[]
    scales = T[]
    for row in axes(constraints, 1)
        scale = sqrt(sum(
            abs2(constraints[row, column]) * kinetic[column]
                for column in axes(constraints, 2)
        ))
        if scale > T(64) * eps(T)
            push!(retained, row)
            push!(scales, scale)
        elseif abs(target[row]) > T(64) * eps(T)
            return (; success = false, values = baseline,
                residual = T(Inf), condition = T(Inf),
                amplification = T(Inf))
        end
    end
    isempty(retained) && return (;
        success = true, values = baseline, residual = zero(T),
        condition = one(T), amplification = one(T),
    )
    A = Matrix(constraints[retained, :])
    d = target[retained]
    for row in axes(A, 1)
        A[row, :] ./= scales[row]
        d[row] /= scales[row]
    end
    gram = (A .* transpose(kinetic)) * transpose(A)
    decomposition = eigen(Symmetric(gram))
    largest = maximum(decomposition.values; init = zero(T))
    cutoff = T(rank_tolerance) * max(largest, one(T))
    retained_modes = findall(>(cutoff), decomposition.values)
    isempty(retained_modes) && return (;
        success = false, values = baseline, residual = T(Inf),
        condition = T(Inf), amplification = T(Inf),
    )
    vectors = decomposition.vectors[:, retained_modes]
    values = decomposition.values[retained_modes]
    correction_rhs = A * baseline - d
    multiplier = vectors * (
        (transpose(vectors) * correction_rhs) ./ values
    )
    result = baseline .- kinetic .* (transpose(A) * multiplier)
    residual = norm(A * result - d, Inf)
    condition = largest / minimum(values)
    baseline_norm = sqrt(sum(abs2(baseline[index]) / kinetic[index]
        for index in eachindex(baseline)))
    correction_norm = sqrt(sum(
        abs2(result[index] - baseline[index]) / kinetic[index]
            for index in eachindex(result)
    ))
    amplification = correction_norm / max(baseline_norm, sqrt(eps(T)))
    residual_limit = max(T(128) * eps(T), T(10) * T(rank_tolerance))
    condition_limit = inv(sqrt(T(rank_tolerance)))
    success = residual <= residual_limit && condition <= condition_limit &&
        amplification <= T(amplification_limit)
    return (; success, values = result, residual, condition, amplification)
end

"""Best local first-order commutator subject to exact normal moments.

Rows of `tangent` are normalised in the kinetic dual norm and the returned
`residual` is their largest response divided by the kinetic norm of the
force column.  It is therefore a dimensionless angle between the multiplier
reaction and the tangential observation.  The solve keeps `moments*q=target`
to roundoff and minimises

    ||tangent*q||² + alpha*||q-baseline||²_inv(K),

with `alpha=O(tolerance²)`.  Accepting `residual <= C*h` gives the requested
first-order commuting relation without weakening normal consistency or the
exact adjoint pairing in the projection.
"""
function _local_first_order_commuting_fit(
        baseline, tangent, moments, target, kinetic;
        tolerance, rank_tolerance, amplification_limit,
    )
    T = eltype(baseline)
    tangent_rows = Int[]
    tangent_scales = T[]
    for row in axes(tangent, 1)
        scale = sqrt(sum(
            abs2(tangent[row, column]) * kinetic[column]
                for column in axes(tangent, 2)
        ))
        scale > T(64) * eps(T) || continue
        push!(tangent_rows, row)
        push!(tangent_scales, scale)
    end
    A = Matrix(tangent[tangent_rows, :])
    for row in axes(A, 1)
        A[row, :] ./= tangent_scales[row]
    end

    moment_rows = Int[]
    moment_scales = T[]
    for row in axes(moments, 1)
        scale = sqrt(sum(
            abs2(moments[row, column]) * kinetic[column]
                for column in axes(moments, 2)
        ))
        if scale > T(64) * eps(T)
            push!(moment_rows, row)
            push!(moment_scales, scale)
        elseif abs(target[row]) > T(64) * eps(T)
            return (; success = false, values = baseline,
                residual = T(Inf), amplification = T(Inf))
        end
    end
    M = Matrix(moments[moment_rows, :])
    d = target[moment_rows]
    for row in axes(M, 1)
        M[row, :] ./= moment_scales[row]
        d[row] /= moment_scales[row]
    end

    inverse_kinetic = inv.(kinetic)
    alpha = max((T(tolerance) / T(10))^2, T(128) * eps(T))
    hessian = transpose(A) * A + Diagonal(alpha .* inverse_kinetic)
    right_hand_side = alpha .* inverse_kinetic .* baseline
    number_of_unknowns = length(baseline)
    number_of_moments = size(M, 1)
    system = [
        hessian transpose(M)
        M zeros(T, number_of_moments, number_of_moments)
    ]
    solution = try
        system \ vcat(right_hand_side, d)
    catch
        return (; success = false, values = baseline,
            residual = T(Inf), amplification = T(Inf))
    end
    result = solution[1:number_of_unknowns]
    all(isfinite, result) || return (;
        success = false, values = baseline,
        residual = T(Inf), amplification = T(Inf),
    )
    force_norm = sqrt(sum(
        abs2(result[index]) / kinetic[index] for index in eachindex(result)
    ))
    residual = isempty(tangent_rows) ? zero(T) :
        norm(A * result, Inf) / max(force_norm, sqrt(eps(T)))
    moment_residual = isempty(moment_rows) ? zero(T) :
        norm(M * result - d, Inf)
    baseline_norm = sqrt(sum(
        abs2(baseline[index]) / kinetic[index]
            for index in eachindex(baseline)
    ))
    correction_norm = sqrt(sum(
        abs2(result[index] - baseline[index]) / kinetic[index]
            for index in eachindex(result)
    ))
    amplification = correction_norm / max(baseline_norm, sqrt(eps(T)))
    moment_limit = max(T(256) * eps(T), T(10) * T(rank_tolerance))
    success = residual <= T(tolerance) && moment_residual <= moment_limit &&
        amplification <= T(amplification_limit)
    return (; success, values = result, residual, amplification)
end

"""Fit one pressure-compatible tangential observation on a local patch.

The constant and affine Cartesian velocity moments are imposed exactly.  In
their nullspace, a regularised least-squares solve fits the two production
pressure responses generated by the affine pressures `p=x` and `p=y`.  The
accepted pressure residual is scaled by the aperture interface measure, so
`residual <= C*h` is a first-order pointwise certificate although `T` stores
the integrated tangential observation.
"""
function _local_pressure_tangent_fit(
        baseline, moments, moment_target, pressure_response,
        pressure_target;
        interface_measure, inverse_density_scale, local_h,
        rank_tolerance, amplification_limit, accuracy_factor,
    )
    T = eltype(baseline)
    metric = ones(T, length(baseline))
    consistent = _local_commuting_fit(
        baseline, moments, moment_target, metric;
        rank_tolerance, amplification_limit,
    )
    consistent.success || return (;
        success = false, values = baseline, residual = T(Inf),
        amplification = T(Inf),
    )

    moment_matrix = Matrix(moments)
    moment_scale = max(norm(moment_matrix), one(T))
    null_basis = nullspace(
        moment_matrix;
        atol = T(rank_tolerance) * moment_scale,
    )
    result = consistent.values
    if !isempty(null_basis)
        reduced_pressure = pressure_response * null_basis
        pressure_rhs = pressure_target -
            pressure_response * consistent.values
        alpha = max(
            (T(inverse_density_scale) * T(local_h))^2,
            T(128) * eps(T),
        )
        reduced_matrix = transpose(reduced_pressure) * reduced_pressure +
            alpha * I
        reduced_rhs = transpose(reduced_pressure) * pressure_rhs +
            alpha * transpose(null_basis) *
            (baseline - consistent.values)
        coefficients = try
            reduced_matrix \ reduced_rhs
        catch
            return (;
                success = false, values = baseline, residual = T(Inf),
                amplification = T(Inf),
            )
        end
        result = consistent.values + null_basis * coefficients
    end
    all(isfinite, result) || return (;
        success = false, values = baseline, residual = T(Inf),
        amplification = T(Inf),
    )

    pressure_scale = max(
        T(interface_measure) * T(inverse_density_scale), sqrt(eps(T)),
    )
    residual = norm(
        pressure_response * result - pressure_target, Inf,
    ) / pressure_scale
    baseline_norm = norm(baseline)
    amplification = norm(result - baseline) /
        max(baseline_norm, sqrt(eps(T)))
    success = residual <= T(accuracy_factor) * T(local_h) &&
        amplification <= T(amplification_limit)
    return (; success, values = result, residual, amplification)
end

"""Construct `T` from the production pressure block before constructing Q.

Only mesh coordinates, phase fractions/apertures, densities, and the
integrated aperture Gauss vector enter this construction.  No PLIC normal or
interface endpoint is used.  Patch growth is local in the interface-row
graph and stops as soon as exact affine velocity moments, an `O(h)` pressure
certificate, and the configured amplification bound all hold.
"""
function _local_pressure_compatible_tangent(
        solver::MimeticProjectionSolver{T}, data, full_active_faces,
        relative_diagonal, reconstruction, constant_samples, jump_space,
    ) where {T}
    mesh = solver.interface.mesh
    baseline = reconstruction.tangent
    normal = reconstruction.normal
    row_count, face_count = size(baseline)
    row_count == 0 && return (;
        tangent = baseline, resolved = falses(0),
        pressure_residual = zero(T), compatible_count = 0,
        fallback_count = 0,
    )

    baseline_supports = _local_sparse_row_supports(baseline)
    normal_supports = _local_sparse_row_supports(normal)
    graph_supports = [
        sort!(unique!(vcat(
            baseline_supports[row], normal_supports[row],
        ))) for row in 1:row_count
    ]
    rows_at_column = _local_rows_at_columns(graph_supports, face_count)
    tangent_moments = baseline * constant_samples
    pressure_force = zeros(T, length(full_active_faces), 2)
    for (local_face, global_face) in pairs(full_active_faces)
        face = mesh.faces[global_face]
        displacement = _face_displacement(mesh, face)
        coefficient = data.coupling[global_face] * T(face.area)
        pressure_force[local_face, 1] = coefficient * displacement[1]
        pressure_force[local_face, 2] = coefficient * displacement[2]
    end
    reduced_force = jump_space.identity ? pressure_force :
        transpose(jump_space.extension) * pressure_force
    pressure_linear = reduced_force ./ reshape(relative_diagonal, :, 1)
    inverse_density_jump = inv(solver.density[2]) -
        inv(solver.density[1])
    inverse_density_scale = max(
        inv(solver.density[1]), inv(solver.density[2]),
    )
    references = _cell_references(mesh)
    fitted_columns = Vector{Tuple{Vector{Int}, Vector{T}}}(undef, row_count)
    resolved = falses(row_count)
    maximum_residual = zero(T)

    for row in 1:row_count
        reference = references[reconstruction.candidate_cells[row]]
        centre = cellcenter(
            mesh, reference.block, reference.i, reference.j,
        )
        spacing = mesh.blocks[reference.block].spacing
        local_h = max(spacing...) / max(mesh.extent...)
        radius = solver.commuting_patch_radius
        previous_patch_size = 0
        accepted = false
        while true
            patch_rows = _local_patch_rows(
                row, radius, graph_supports, rows_at_column,
            )
            support = Int[]
            for patch_row in patch_rows
                append!(support, graph_supports[patch_row])
            end
            sort!(unique!(support))
            baseline_values = vec(Array(baseline[row:row, support]))
            constant_moments = transpose(
                constant_samples[support, :],
            )
            affine_moments = zeros(T, 4, length(support))
            for (local_column, active_face) in pairs(support)
                face = mesh.faces[jump_space.active_faces[active_face]]
                dx = _wrapped_delta(
                    face.center[1] - centre[1], mesh.extent[1],
                    mesh.periodic[1],
                ) / spacing[1]
                dy = _wrapped_delta(
                    face.center[2] - centre[2], mesh.extent[2],
                    mesh.periodic[2],
                ) / spacing[2]
                offset = 2 * (face.axis - 1)
                affine_moments[offset + 1, local_column] = dx
                affine_moments[offset + 2, local_column] = dy
            end
            moments = vcat(constant_moments, affine_moments)
            moment_target = vcat(
                vec(tangent_moments[row, :]), zeros(T, 4),
            )
            pressure_response = transpose(hcat(
                pressure_linear[support, 1],
                pressure_linear[support, 2],
            ))
            pressure_target = inverse_density_jump .* vec(
                tangent_moments[row, :],
            )
            fit = _local_pressure_tangent_fit(
                baseline_values, moments, moment_target,
                pressure_response, pressure_target;
                interface_measure = reconstruction.candidate_scales[row],
                inverse_density_scale, local_h,
                rank_tolerance = solver.rank_tolerance,
                amplification_limit =
                    solver.commuting_amplification_limit,
                accuracy_factor = solver.commuting_accuracy_factor,
            )
            if fit.success
                fitted_columns[row] = (support, fit.values)
                resolved[row] = true
                maximum_residual = max(maximum_residual, fit.residual)
                accepted = true
                break
            end
            saturated = length(patch_rows) == previous_patch_size
            if radius < solver.commuting_max_patch_radius && !saturated
                previous_patch_size = length(patch_rows)
                radius += 1
                continue
            end
            break
        end
        accepted && continue
        support = baseline_supports[row]
        fitted_columns[row] = (
            support, vec(Array(baseline[row:row, support])),
        )
    end

    tangent_rows = Int[]
    tangent_columns = Int[]
    tangent_values = T[]
    coefficient_tolerance = T(64) * eps(T)
    for (row, (support, values)) in pairs(fitted_columns)
        for (local_index, value) in pairs(values)
            abs(value) > coefficient_tolerance || continue
            push!(tangent_rows, row)
            push!(tangent_columns, support[local_index])
            push!(tangent_values, value)
        end
    end
    tangent = sparse(
        tangent_rows, tangent_columns, tangent_values,
        row_count, face_count,
    )
    return (;
        tangent, resolved,
        pressure_residual = maximum_residual,
        compatible_count = count(resolved),
        fallback_count = count(!, resolved),
    )
end

"""Construct the certified commuting aperture-normal constraint.

The two hard invariants are:

* `Z'Q = (N0*Z)'`, so `N=Q'` reproduces the baseline normal moments;
* the KKT system uses exactly `Q` and `Q'`, preserving symmetry, momentum,
  and the kinetic projection identity for every accepted or fallback row.

The construction first seeks exact commutativity,
`T*inv(K_xi)*Q = 0`.  If that is incompatible with a compact patch, it keeps
the normal moments exact and accepts the best local kinetic fit when its
dimensionless leakage is `O(h)`.  Patch growth can reach the whole connected
row component when necessary.  If a column still cannot be certified, its
affected tangential rows are downgraded and the construction is repeated.
This monotone loop always terminates: with no certified tangential rows, the
consistent aperture baseline `N_a` is feasible without modification.
"""
function _local_commuting_constraint_operator(
        solver::MimeticProjectionSolver{T}, data, full_active_faces,
        relative_diagonal, jump_space,
    ) where {T}
    full_reconstruction = _local_aperture_normal_tangent_reconstruction(
        solver, data, full_active_faces,
    )
    reconstruction = if jump_space.identity
        full_reconstruction
    else
        # Substitute xi = E*xi_hat before constructing T and Q=N'. Both
        # observations therefore see the same constant extrapolation as the
        # production pressure and kinetic blocks.
        reduced_normal =
            full_reconstruction.normal * jump_space.extension
        reduced_tangent =
            full_reconstruction.tangent * jump_space.extension
        normal_supports = _local_sparse_row_supports(reduced_normal)
        retained_rows = findall(row -> !isempty(row), normal_supports)
        (;
            normal = reduced_normal[retained_rows, :],
            tangent = reduced_tangent[retained_rows, :],
            candidate_cells =
                full_reconstruction.candidate_cells[retained_rows],
            candidate_scales =
                full_reconstruction.candidate_scales[retained_rows],
            tangent_resolved =
                full_reconstruction.tangent_resolved[retained_rows],
            underresolved = full_reconstruction.underresolved +
                size(reduced_normal, 1) - length(retained_rows),
        )
    end
    baseline = reconstruction.normal
    row_count, face_count = size(baseline)
    row_count == 0 && return (;
        constraint = baseline, candidate_constraint = baseline,
        candidate_locations = reconstruction.candidate_cells,
        constraint_locations = reconstruction.candidate_cells,
        constraint_scales = reconstruction.candidate_scales,
        candidate_scales = reconstruction.candidate_scales,
        candidate_count = 0, dependent_count = 0,
        underresolved = reconstruction.underresolved,
        commuting_count = 0, fallback_count = 0,
        exact_commuting_count = 0, approximate_commuting_count = 0,
        commuting_residual = zero(T),
        pressure_commuting_count = 0, pressure_fallback_count = 0,
        pressure_commuting_residual = zero(T),
    )

    constant_samples = zeros(T, face_count, 2)
    mesh = solver.interface.mesh
    for (local_face, global_face) in pairs(jump_space.active_faces)
        constant_samples[local_face, mesh.faces[global_face].axis] = one(T)
    end
    pressure_tangent = _local_pressure_compatible_tangent(
        solver, data, full_active_faces, relative_diagonal, reconstruction,
        constant_samples, jump_space,
    )
    tangent = pressure_tangent.tangent
    normal_moments = baseline * constant_samples
    normal_supports = _local_sparse_row_supports(baseline)
    tangent_supports = _local_sparse_row_supports(tangent)
    combined_supports = [
        sort!(unique!(vcat(normal_supports[row], tangent_supports[row])))
            for row in 1:row_count
    ]
    rows_at_column = _local_rows_at_columns(combined_supports, face_count)
    tangent_rows_at_column = _local_rows_at_columns(
        tangent_supports, face_count,
    )
    resolved = BitVector(
        reconstruction.tangent_resolved .& pressure_tangent.resolved,
    )
    final_columns = Vector{Tuple{Vector{Int}, Vector{T}}}(undef, row_count)
    column_exact = falses(row_count)
    references = _cell_references(mesh)

    for _ in 1:(row_count + 1)
        failed_rows = falses(row_count)
        fill!(column_exact, false)
        for normal_row in 1:row_count
            fitted = false
            last_overlapping = Int[]
            radius = solver.commuting_patch_radius
            previous_patch_size = 0
            while true
                patch_rows = _local_patch_rows(
                    normal_row, radius, combined_supports, rows_at_column,
                )
                support = Int[]
                for row in patch_rows
                    append!(support, combined_supports[row])
                end
                sort!(unique!(support))
                overlapping = Int[]
                for column in support
                    append!(overlapping, tangent_rows_at_column[column])
                end
                sort!(unique!(overlapping))
                filter!(row -> resolved[row], overlapping)
                last_overlapping = overlapping

                tangent_constraints = Matrix(tangent[overlapping, support])
                tangent_constraints .*= transpose(
                    inv.(relative_diagonal[support]),
                )
                moment_constraints = transpose(
                    constant_samples[support, :],
                )
                constraints = vcat(tangent_constraints, moment_constraints)
                target = vcat(
                    zeros(T, length(overlapping)),
                    vec(normal_moments[normal_row, :]),
                )
                baseline_values = vec(Array(
                    baseline[normal_row:normal_row, support],
                ))
                fit = _local_commuting_fit(
                    baseline_values, constraints, target,
                    relative_diagonal[support];
                    rank_tolerance = solver.rank_tolerance,
                    amplification_limit =
                        solver.commuting_amplification_limit,
                )
                if fit.success
                    final_columns[normal_row] = (support, fit.values)
                    column_exact[normal_row] = true
                    fitted = true
                    break
                end
                reference = references[
                    reconstruction.candidate_cells[normal_row]
                ]
                spacing = mesh.blocks[reference.block].spacing
                local_h = max(spacing...) / max(mesh.extent...)
                first_order_tolerance =
                    solver.commuting_accuracy_factor * T(local_h)
                approximate = _local_first_order_commuting_fit(
                    baseline_values, tangent_constraints,
                    moment_constraints,
                    vec(normal_moments[normal_row, :]),
                    relative_diagonal[support];
                    tolerance = first_order_tolerance,
                    rank_tolerance = solver.rank_tolerance,
                    amplification_limit =
                        solver.commuting_amplification_limit,
                )
                if approximate.success
                    final_columns[normal_row] = (
                        support, approximate.values,
                    )
                    fitted = true
                    break
                end
                # The default maximum is component-wide. Stop increasing the
                # graph radius once the connected component is exhausted;
                # this avoids an artificial locality failure while retaining
                # a finite escape for a genuinely infeasible component solve.
                saturated = length(patch_rows) == previous_patch_size
                if radius < solver.commuting_max_patch_radius && !saturated
                    previous_patch_size = length(patch_rows)
                    radius += 1
                    continue
                end
                break
            end
            fitted && continue
            # This column cannot support every currently certified T row.
            # Downgrade precisely those rows; its baseline normal moments are
            # never discarded.
            failed_rows[last_overlapping] .= true
            support = normal_supports[normal_row]
            final_columns[normal_row] = (
                support,
                vec(Array(baseline[normal_row:normal_row, support])),
            )
        end
        any(failed_rows) || break
        resolved[failed_rows] .= false
    end

    q_rows = Int[]
    q_columns = Int[]
    q_values = T[]
    coefficient_tolerance = T(64) * eps(T)
    for (normal_row, (support, values)) in pairs(final_columns)
        for (local_index, value) in pairs(values)
            abs(value) > coefficient_tolerance || continue
            push!(q_rows, support[local_index])
            push!(q_columns, normal_row)
            push!(q_values, value)
        end
    end
    Q = sparse(q_rows, q_columns, q_values, face_count, row_count)
    candidate = sparse(transpose(Q))
    commuting_residual = if any(resolved)
        product = tangent[resolved, :] *
            spdiagm(0 => inv.(relative_diagonal)) * Q
        tangent_norms = sqrt.(diag(
            tangent[resolved, :] *
                spdiagm(0 => inv.(relative_diagonal)) *
                transpose(tangent[resolved, :]),
        ))
        force_norms = sqrt.(diag(
            transpose(Q) * spdiagm(0 => inv.(relative_diagonal)) * Q,
        ))
        maximum(
            (
                abs(product[row, column]) /
                    max(
                        tangent_norms[row] * force_norms[column],
                        sqrt(eps(T)),
                    )
                for row in axes(product, 1), column in axes(product, 2)
            );
            init = zero(T),
        )
    else
        zero(T)
    end
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    return (;
        constraint = candidate[retained, :],
        candidate_constraint = candidate,
        candidate_locations = reconstruction.candidate_cells,
        constraint_locations = reconstruction.candidate_cells[retained],
        constraint_scales = reconstruction.candidate_scales[retained],
        candidate_scales = reconstruction.candidate_scales,
        candidate_count = row_count,
        dependent_count = row_count - length(retained),
        underresolved = reconstruction.underresolved + count(!, resolved),
        commuting_count = count(resolved),
        exact_commuting_count = all(column_exact) ? count(resolved) : 0,
        approximate_commuting_count = all(column_exact) ? 0 : count(resolved),
        fallback_count = count(!, resolved),
        commuting_residual,
        pressure_commuting_count = pressure_tangent.compatible_count,
        pressure_fallback_count = pressure_tangent.fallback_count,
        pressure_commuting_residual = pressure_tangent.pressure_residual,
    )
end

"""Assemble the original centred aperture-divergence jump interpolant."""
function _mimetic_divergence_constraint_operator(
        solver::MimeticProjectionSolver{T}, data, active_faces,
        relative_diagonal, incidence, pressure_operator,
    ) where {T}
    mesh = solver.interface.mesh
    active_cells = _mimetic_active_cells(solver.interface)
    nx, ny = mesh.block_size
    eligible = falses(ncells(mesh))
    coefficients = data.aperture_jump[active_faces]
    for (local_index, face_index) in pairs(active_faces)
        iszero(coefficients[local_index]) && continue
        face = mesh.faces[face_index]
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        eligible[minus] = active_cells[minus]
        eligible[plus] = active_cells[plus]
    end
    candidate_cells = findall(eligible)
    candidate = isempty(candidate_cells) ?
        spzeros(T, 0, length(active_faces)) :
        transpose(incidence[active_faces, candidate_cells]) *
            spdiagm(0 => coefficients)
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    scales = pressure_operator.volumes[candidate_cells]
    return (;
        constraint = candidate[retained, :],
        candidate_constraint = candidate,
        candidate_locations = candidate_cells,
        constraint_locations = candidate_cells[retained],
        constraint_scales = scales[retained],
        candidate_scales = scales,
        candidate_count = length(candidate_cells),
        dependent_count = length(candidate_cells) - length(retained),
        underresolved = 0,
    )
end

function _local_constraint_operator(
        solver, data, active_faces, relative_diagonal, incidence,
        pressure_operator; jump_space = nothing,
    )
    if solver.jump_interpolant === :divergence
        return _mimetic_divergence_constraint_operator(
            solver, data, active_faces, relative_diagonal, incidence,
            pressure_operator,
        )
    end
    if solver.jump_interpolant === :cell
        return _local_cell_constraint_operator(
            solver, data, active_faces, relative_diagonal,
        )
    end
    if solver.jump_interpolant === :commuting
        return _local_commuting_constraint_operator(
            solver, data, jump_space.full_active_faces,
            relative_diagonal, jump_space,
        )
    end
    return _local_face_constraint_operator(
        solver, data, active_faces, relative_diagonal,
    )
end

function _assemble_mimetic_system!(
        solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, dt,
        velocity::TwoPhaseVelocity,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    data = _mimetic_face_data(solver, workspace)
    pressure_operator = _mimetic_pressure_operator(solver)
    incidence = _mimetic_incidence(mesh, T)
    active_cells = _mimetic_active_cells(interface)
    open_weights = isnothing(interface.geometry) ?
        ones(T, length(mesh.faces)) : copy(interface.geometry.aperture)
    pressure_indices = _gauge_free_component_rows(
        mesh, active_cells, open_weights,
    )
    full_active_faces, mass_deactivated, embedded_deactivated =
        _local_active_faces(solver, data)
    active_set = Set(full_active_faces)
    inactive_faces = filter(
        face_index -> !(face_index in active_set), data.mixed,
    )
    pressure_positive = pressure_operator.stiffness[
        pressure_indices, pressure_indices,
    ]
    incidence_pressure = incidence[:, pressure_indices]
    coupling = data.coupling[full_active_faces]
    full_pressure_jump =
        -transpose(incidence_pressure[full_active_faces, :]) *
            spdiagm(0 => coupling)
    all_face_dual_volumes = isnothing(interface.geometry) ?
        _uncut_face_dual_volumes(mesh, T) :
        _geometry_face_dual_volumes(interface.geometry)
    full_face_dual_volumes = all_face_dual_volumes[full_active_faces]
    full_relative_diagonal =
        data.relative_mass[full_active_faces] .* full_face_dual_volumes
    all(>(zero(T)), full_relative_diagonal) || throw(
        ArgumentError("active jump faces require positive phase masses"),
    )
    jump_space = _local_commuting_jump_space(
        solver, data, full_active_faces, full_relative_diagonal,
        full_face_dual_volumes,
    )
    active_faces = jump_space.active_faces
    relative_diagonal = jump_space.relative_diagonal
    face_dual_volumes = jump_space.face_dual_volumes
    pressure_jump = full_pressure_jump * jump_space.extension
    local_operator = _local_constraint_operator(
        solver, data, active_faces, relative_diagonal, incidence,
        pressure_operator; jump_space,
    )
    commuting_count = hasproperty(local_operator, :commuting_count) ?
        local_operator.commuting_count : 0
    exact_commuting_count =
        hasproperty(local_operator, :exact_commuting_count) ?
        local_operator.exact_commuting_count : 0
    approximate_commuting_count =
        hasproperty(local_operator, :approximate_commuting_count) ?
        local_operator.approximate_commuting_count : 0
    fallback_count = hasproperty(local_operator, :fallback_count) ?
        local_operator.fallback_count : 0
    commuting_residual = hasproperty(local_operator, :commuting_residual) ?
        local_operator.commuting_residual : zero(T)
    pressure_commuting_count =
        hasproperty(local_operator, :pressure_commuting_count) ?
        local_operator.pressure_commuting_count : 0
    pressure_fallback_count =
        hasproperty(local_operator, :pressure_fallback_count) ?
        local_operator.pressure_fallback_count : 0
    pressure_commuting_residual =
        hasproperty(local_operator, :pressure_commuting_residual) ?
        local_operator.pressure_commuting_residual : zero(T)
    constraint = local_operator.constraint
    multiplier_count = size(constraint, 1)
    zero_pl = spzeros(T, length(pressure_indices), multiplier_count)
    matrix = [
        -pressure_positive pressure_jump zero_pl
        transpose(pressure_jump) spdiagm(0 => relative_diagonal) transpose(constraint)
        transpose(zero_pl) constraint spzeros(T, multiplier_count, multiplier_count)
    ]
    block_preconditioner = _local_block_preconditioner(
        pressure_positive, relative_diagonal, constraint, T,
    )

    mixture = data.liquid_aperture .* velocity[1].interior .+
        data.gas_aperture .* velocity[2].interior
    jump_velocity = velocity[2].interior .- velocity[1].interior
    # On a face removed by the mass/geometry active set, constrain the final
    # jump to zero.  Treating its correction as known keeps the feasible
    # velocity space homogeneous, which is essential for the pressure-work
    # (and hence energy) identity.  Merely setting xi=0 would leave an affine
    # divergence constraint and destroy orthogonality.
    inactive_jump_gradient = jump_velocity[inactive_faces] ./ T(dt)
    inactive_pressure_jump = isempty(inactive_faces) ?
        spzeros(T, length(pressure_indices), 0) :
        -transpose(incidence_pressure[inactive_faces, :]) *
            spdiagm(0 => data.coupling[inactive_faces])
    boundary_mixture, boundary_normal =
        _mimetic_boundary_constraints(interface, velocity)
    pressure_rhs = (
        -transpose(incidence_pressure) * mixture .+
            boundary_mixture[pressure_indices]
    ) ./ T(dt)
    pressure_rhs .-= inactive_pressure_jump * inactive_jump_gradient
    constraint_boundary = solver.jump_interpolant === :divergence ?
        boundary_normal[local_operator.constraint_locations] :
        zeros(T, multiplier_count)
    candidate_boundary = solver.jump_interpolant === :divergence ?
        boundary_normal[local_operator.candidate_locations] :
        zeros(T, local_operator.candidate_count)
    # The reduced coordinate of an arbitrary physical velocity jump is its
    # K_xi-orthogonal projection onto range(E). With xi=E*xi_hat this gives
    # j_hat=(E'K_xi E)^(-1)E'K_xi*j, so that
    # j_hat_new = j_hat_old-dt*xi_hat. Using donor-face samples here would
    # break that identity and hence the discrete energy proof.
    reduced_jump_velocity = isempty(active_faces) ? T[] :
        (transpose(jump_space.extension) *
            (full_relative_diagonal .*
                jump_velocity[full_active_faces])) ./ relative_diagonal
    constraint_rhs = (
        constraint * reduced_jump_velocity .- constraint_boundary
    ) ./ T(dt)
    right_hand_side = vcat(
        pressure_rhs, zeros(T, length(active_faces)), constraint_rhs,
    )
    return (;
        data, pressure_operator, incidence, matrix, block_preconditioner,
        right_hand_side, pressure_indices, active_faces, constraint,
        candidate_constraint = local_operator.candidate_constraint,
        candidate_boundary,
        candidate_scales = local_operator.candidate_scales,
        inactive_faces, inactive_jump_gradient,
        constraint_locations = local_operator.constraint_locations,
        constraint_scales = local_operator.constraint_scales,
        candidate_count = local_operator.candidate_count,
        dependent_count = local_operator.dependent_count,
        underresolved = local_operator.underresolved,
        commuting_count, exact_commuting_count,
        approximate_commuting_count, fallback_count, commuting_residual,
        pressure_commuting_count, pressure_fallback_count,
        pressure_commuting_residual,
        mass_deactivated, embedded_deactivated,
        relative_diagonal, face_dual_volumes,
        full_relative_diagonal,
        jump_residual_scales = jump_space.residual_scales,
        full_active_faces,
        jump_extension = jump_space.extension,
        sliver_faces = jump_space.sliver_faces,
        sliver_donor_faces = jump_space.sliver_donor_faces,
        sliver_extrapolated = jump_space.extrapolated_count,
        sliver_zeroed = jump_space.zeroed_count,
    )
end

function _local_mimetic_residual_defects(residual, assembled, dt)
    T = eltype(residual)
    np = length(assembled.pressure_indices)
    nx = length(assembled.active_faces)
    nl = size(assembled.constraint, 1)
    volumes = assembled.pressure_operator.volumes
    pressure = zero(T)
    for local_index in 1:np
        cell = @inbounds assembled.pressure_indices[local_index]
        @inbounds pressure = max(
            pressure, abs(residual[local_index]) / volumes[cell],
        )
    end
    tangent = zero(T)
    for local_index in 1:nx
        scale = @inbounds assembled.jump_residual_scales[local_index]
        @inbounds tangent = max(
            tangent, abs(residual[np + local_index]) / scale,
        )
    end
    normal = zero(T)
    for local_index in 1:nl
        scale = @inbounds assembled.constraint_scales[local_index]
        @inbounds normal = max(
            normal, abs(residual[np + nx + local_index]) / scale,
        )
    end
    return (;
        pressure = T(dt)^2 * pressure,
        tangent = T(dt)^2 * tangent,
        normal = T(dt) * normal,
    )
end

function _local_mimetic_preconditioner(
        solver::MimeticProjectionSolver, block_preconditioner,
    )
    specification = solver.preconditioner
    specification === nothing && return nothing
    if specification === :cholesky
        isempty(solver.preconditioner_options) || throw(
            ArgumentError(
                "preconditioner_options are not used by exact Cholesky",
            ),
        )
        return ExternalPreconditioner(
            :cholesky,
            cholesky(Symmetric(block_preconditioner, :L)),
            block_preconditioner,
        )
    end
    return _construct_matrix_preconditioner(
        specification, block_preconditioner;
        symmetric = true, singular = false,
        options = solver.preconditioner_options,
    )
end

"""Reconstruct both phase pressure gradients from the latest local solve."""
function mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::MimeticProjectionSolver{T},
        workspace::MimeticProjectionWorkspace{T}, pressure_operator,
        data,
    ) where {T}
    _validate_two_phase_velocity(destination, solver.interface.mesh, T)
    elliptic = EllipticWorkspace(pressure_operator)
    _mimetic_weighted_gradient!(
        workspace.pressure_gradient, pressure, pressure_operator, elliptic,
        solver.interface.geometry,
    )
    gradient!(
        workspace.base_gradient, pressure,
        pressure_operator.gradient_operator,
    )
    for index in eachindex(solver.interface.mesh.faces)
        mass = @inbounds data.total_mass[index]
        jump = @inbounds workspace.jump[index]
        common = @inbounds workspace.pressure_gradient.interior[index]
        @inbounds begin
            destination[1].interior[index] = common +
                data.zeta_liquid[index] / mass * jump
            destination[2].interior[index] = common +
                data.zeta_gas[index] / mass * jump
        end
    end
    fill!(destination[1].boundary, zero(T))
    fill!(destination[2].boundary, zero(T))
    return destination
end

function mimetic_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::MimeticProjectionSolver,
        workspace::MimeticProjectionWorkspace,
    )
    operator = _mimetic_pressure_operator(solver)
    data = _mimetic_face_data(solver, workspace)
    return mimetic_gradient!(
        destination, pressure, solver, workspace, operator, data,
    )
end

"""
    project!(velocity, pressure, solver::MimeticProjectionSolver, dt,
             workspace; tolerance, max_iterations)

Project onto mixture incompressibility and the selected centred, face-local,
or cell-local jump condition. Independent constraint rows and a phase-mass
active set make the three-block KKT system nonsingular; its block
preconditioner contains the multiplier Schur approximation `H*inv(K_xi)*H'`.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::MimeticProjectionSolver{T}, dt,
        workspace::MimeticProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500,
    ) where {T}
    interface = solver.interface
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    pressure.mesh === interface.mesh || throw(
        ArgumentError("mimetic projection arguments must share a mesh"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    tolerance > zero(tolerance) || throw(
        ArgumentError("tolerance must be strictly positive"),
    )
    max_iterations > 0 || throw(
        ArgumentError("max_iterations must be strictly positive"),
    )
    assembled = _assemble_mimetic_system!(
        solver, workspace, dt, velocity,
    )
    matrix = assembled.matrix
    right_hand_side = assembled.right_hand_side
    symmetry_error = isempty(matrix) ? zero(T) :
        maximum(abs, matrix - transpose(matrix))
    initial_residual = norm(right_hand_side)
    solution = zeros(T, length(right_hand_side))
    iterations = 0
    preconditioner = nothing
    if !isempty(solution) && initial_residual > zero(T)
        preconditioner = _local_mimetic_preconditioner(
            solver, assembled.block_preconditioner,
        )
        correction_rhs = copy(right_hand_side)
        for refinement in 1:8
            remaining = max_iterations - iterations
            remaining > 0 || break
            defects = _local_mimetic_residual_defects(
                correction_rhs, assembled, dt,
            )
            maximum(defects) <= T(tolerance) && break
            krylov = Krylov.MinresWorkspace(matrix, correction_rhs)
            requested = min(
                T(tolerance), T(tolerance) / max(maximum(defects), eps(T)),
            )
            if isnothing(preconditioner)
                Krylov.minres!(
                    krylov, matrix, correction_rhs;
                    atol = zero(T), rtol = requested,
                    etol = zero(T), conlim = zero(T), itmax = remaining,
                )
            else
                Krylov.minres!(
                    krylov, matrix, correction_rhs;
                    M = preconditioner, ldiv = true,
                    atol = zero(T), rtol = requested,
                    etol = zero(T), conlim = zero(T), itmax = remaining,
                )
            end
            solution .+= Krylov.solution(krylov)
            iterations += Krylov.iteration_count(krylov)
            mul!(correction_rhs, matrix, solution)
            correction_rhs .= right_hand_side .- correction_rhs
        end
    end
    residual = matrix * solution - right_hand_side
    final_residual = norm(residual)
    residual_defects = _local_mimetic_residual_defects(
        residual, assembled, dt,
    )

    np = length(assembled.pressure_indices)
    nx = length(assembled.active_faces)
    nl = size(assembled.constraint, 1)
    pressure_values = view(solution, 1:np)
    jump_values = view(solution, (np + 1):(np + nx))
    _copy_mimetic_cell_solution!(
        pressure, zip(assembled.pressure_indices, pressure_values),
    )
    fill!(workspace.jump, zero(T))
    workspace.jump[assembled.inactive_faces] .=
        assembled.inactive_jump_gradient
    workspace.jump[assembled.full_active_faces] .=
        assembled.jump_extension * jump_values

    gradients = (workspace.liquid_gradient, workspace.gas_gradient)
    mimetic_gradient!(
        gradients, pressure, solver, workspace,
        assembled.pressure_operator, assembled.data,
    )
    for phase in 1:2
        for index in eachindex(velocity[phase].interior)
            @inbounds velocity[phase].interior[index] -=
                T(dt) * gradients[phase].interior[index]
        end
    end

    final_liquid = velocity[1].interior
    final_gas = velocity[2].interior
    orthogonality = zero(T)
    pressure_momentum = zeros(T, 2)
    identity_error = zero(T)
    geometry = interface.geometry
    pressure_jump =
        assembled.pressure_operator.gradient_operator.pressure_jump
    for (index, face) in pairs(interface.mesh.faces)
        axis = face.axis
        @inbounds begin
            liquid_mass = assembled.data.liquid_mass[index]
            gas_mass = assembled.data.gas_mass[index]
            liquid_gradient = gradients[1].interior[index]
            gas_gradient = gradients[2].interior[index]
            final_common = (
                liquid_mass * final_liquid[index] +
                    gas_mass * final_gas[index]
            ) / (liquid_mass + gas_mass)
            open_aperture = isnothing(geometry) ? one(T) :
                geometry.aperture[index]
            orthogonality +=
                final_common * open_aperture * pressure_jump[index]
            weighted_gradient =
                liquid_mass * liquid_gradient + gas_mass * gas_gradient
            pressure_momentum[axis] -= T(dt) * pressure_jump[index]
            identity_error = max(
                identity_error, abs(
                    weighted_gradient -
                        (liquid_mass + gas_mass) *
                        workspace.pressure_gradient.interior[index]
                ),
            )
        end
    end
    all_face_dual_volumes = isnothing(interface.geometry) ?
        _uncut_face_dual_volumes(interface.mesh, T) :
        _geometry_face_dual_volumes(interface.geometry)
    for face_index in assembled.data.mixed
        @inbounds orthogonality +=
            (final_gas[face_index] - final_liquid[face_index]) *
            assembled.data.relative_mass[face_index] *
            all_face_dual_volumes[face_index] *
            workspace.jump[face_index]
    end
    energy_error = abs(2T(dt) * orthogonality)
    mixture_constraint, = _mimetic_integrated_constraints(
        assembled.incidence, assembled.data, interface, velocity,
        assembled.active_faces,
    )
    volumes = assembled.pressure_operator.volumes
    mixture_divergence = _mimetic_strong_maximum(
        mixture_constraint, volumes,
    )
    mixture_volume_defect = T(dt) * mixture_divergence
    full_final_jump = final_gas[assembled.full_active_faces] .-
        final_liquid[assembled.full_active_faces]
    final_jump = isempty(assembled.active_faces) ? T[] :
        (transpose(assembled.jump_extension) *
            (assembled.full_relative_diagonal .* full_final_jump)) ./
                assembled.relative_diagonal
    # Assess every resolved local row, including rows omitted from the solve
    # because rank revelation classified them as dependent.  This prevents
    # the rank filter from hiding a loss of local consistency.
    local_residual = assembled.candidate_constraint * final_jump .-
        assembled.candidate_boundary
    normal_velocity_defect = zero(T)
    for index in eachindex(local_residual)
        @inbounds normal_velocity_defect = max(
            normal_velocity_defect,
            abs(local_residual[index]) / assembled.candidate_scales[index],
        )
    end
    for face_index in assembled.inactive_faces
        @inbounds normal_velocity_defect = max(
            normal_velocity_defect,
            abs(final_gas[face_index] - final_liquid[face_index]),
        )
    end
    normal_jump_divergence = normal_velocity_defect
    normal_volume_defect = solver.jump_interpolant === :divergence ?
        T(dt) * normal_jump_divergence : normal_velocity_defect
    converged = mixture_volume_defect <= T(tolerance) &&
        normal_volume_defect <= T(tolerance) &&
        residual_defects.tangent <= T(tolerance)
    preconditioner_name = isnothing(preconditioner) ? :none :
        _preconditioner_name(preconditioner)

    workspace.matrix = matrix
    workspace.block_preconditioner = assembled.block_preconditioner
    workspace.constraint_matrix = assembled.constraint
    workspace.right_hand_side = right_hand_side
    workspace.solution = solution
    workspace.pressure_indices = assembled.pressure_indices
    workspace.active_faces = assembled.active_faces
    workspace.mixed_faces = assembled.full_active_faces
    workspace.constraint_locations = assembled.constraint_locations
    workspace.constraint_scales = assembled.constraint_scales
    return MimeticProjectionStats(
        converged, iterations, initial_residual, final_residual,
        symmetry_error, mixture_divergence, mixture_volume_defect,
        normal_jump_divergence, normal_volume_defect,
        normal_velocity_defect, residual_defects.tangent, T(tolerance),
        (pressure_momentum[1], pressure_momentum[2]), identity_error,
        energy_error, abs(orthogonality), np, nx, nl,
        assembled.candidate_count, nl, assembled.dependent_count,
        assembled.underresolved, assembled.commuting_count,
        assembled.exact_commuting_count,
        assembled.approximate_commuting_count,
        assembled.fallback_count, assembled.commuting_residual,
        assembled.pressure_commuting_count,
        assembled.pressure_fallback_count,
        assembled.pressure_commuting_residual,
        assembled.mass_deactivated,
        assembled.embedded_deactivated,
        assembled.sliver_extrapolated, assembled.sliver_zeroed,
        solver.commuting_sliver_epsilon, solver.jump_interpolant,
        solver.interpolation_weights, preconditioner_name,
    )
end
