"""Bounded, local fit of physical normal and tangential jump rows.

This file implements `variant=:joint_smoothed` and `:joint_face`. It keeps the *full*
active jump space and its diagonal kinetic metric.  Only `N` is put into
the symmetric KKT system; `T` is an observation used to choose `N`.

Stationarity then implies

    T*xi = -T*inv(K)*C'*p - T*inv(K)*N'*lambda.

The fit reproduces constant vector jumps exactly in both `T` and `N`,
fits the pressure response on local affine pressure modes, and makes the
second term small for all overlapping multiplier rows.  The two fitting
objectives are soft: the production KKT retains every cross term, so its
momentum and energy identities do not depend on exact commutation.
"""

function _local_joint_prepare!(
        scratch::MimeticJointFitScratch{T}, row_count, face_count,
    ) where {T}
    while length(scratch.rows) < row_count
        push!(scratch.rows, _MimeticJointFitRow(T))
        push!(scratch.graph_supports, Int[])
    end
    while length(scratch.graph_rows_at_column) < face_count
        push!(scratch.graph_rows_at_column, Int[])
        push!(scratch.matrix_rows_at_column, Int[])
    end
    for row in 1:row_count
        empty!(scratch.graph_supports[row])
        patch = scratch.rows[row]
        patch.geometry_fallback = false
        patch.tangent_success = true
        patch.normal_success = true
    end
    for column in 1:face_count
        empty!(scratch.graph_rows_at_column[column])
    end
    resize!(scratch.overlap_lookup, row_count)
    if size(scratch.constant_samples) != (face_count, 2)
        scratch.constant_samples = zeros(T, face_count, 2)
        scratch.common_pressure = zeros(T, face_count, 2)
    else
        fill!(scratch.constant_samples, zero(T))
        fill!(scratch.common_pressure, zero(T))
    end
    return nothing
end

function _local_joint_graph!(
        scratch::MimeticJointFitScratch, normal, tangent, row_count,
    )
    # Preserve normal-then-tangent support ordering before sorting each row.
    for matrix in (normal, tangent)
        matrix_rows = rowvals(matrix)
        for column in axes(matrix, 2)
            for pointer in nzrange(matrix, column)
                push!(scratch.graph_supports[matrix_rows[pointer]], column)
            end
        end
    end
    for row in 1:row_count
        support = scratch.graph_supports[row]
        sort!(unique!(support))
        for column in support
            push!(scratch.graph_rows_at_column[column], row)
        end
    end
    return nothing
end

function _local_joint_patch_support!(
        scratch::MimeticJointFitScratch, row, radius,
    )
    selected = scratch.selected
    frontier, next_frontier = scratch.frontier, scratch.next_frontier
    empty!(selected)
    empty!(frontier)
    push!(selected, row)
    push!(frontier, row)
    for _ in 1:radius
        empty!(next_frontier)
        for patch_row in frontier
            for column in scratch.graph_supports[patch_row]
                for neighbour in scratch.graph_rows_at_column[column]
                    neighbour in selected && continue
                    push!(selected, neighbour)
                    push!(next_frontier, neighbour)
                end
            end
        end
        frontier, next_frontier = next_frontier, frontier
        isempty(frontier) && break
    end
    support = scratch.rows[row].support
    empty!(support)
    for patch_row in selected
        append!(support, scratch.graph_supports[patch_row])
    end
    sort!(unique!(support))
    return support
end

function _local_joint_matrix(
        scratch::MimeticJointFitScratch{T}, row_count, face_count, tangent::Bool,
    ) where {T}
    rows = scratch.matrix_rows
    columns = scratch.matrix_columns
    coefficients = scratch.matrix_coefficients
    empty!(rows)
    empty!(columns)
    empty!(coefficients)
    for row in 1:row_count
        patch = scratch.rows[row]
        values = tangent ? patch.tangent_values : patch.normal_values
        for (position, column) in pairs(patch.support)
            value = values[position]
            abs(value) <= T(64) * eps(T) && continue
            push!(rows, row)
            push!(columns, column)
            push!(coefficients, value)
        end
    end
    return sparse(rows, columns, coefficients, row_count, face_count)
end

function _local_joint_matrix_rows_at_columns!(
        scratch::MimeticJointFitScratch, matrix,
    )
    matrix_rows = rowvals(matrix)
    for column in axes(matrix, 2)
        rows = scratch.matrix_rows_at_column[column]
        empty!(rows)
        for pointer in nzrange(matrix, column)
            push!(rows, matrix_rows[pointer])
        end
    end
    return scratch.matrix_rows_at_column
end

"""Regularised local fit in the exact constant-moment nullspace.

Each pressure/commutator observation is divided by its row norm before
weighting.  The cap is relative to the geometric row, which prevents the
exact pressure moment equations from creating the very large coefficients
seen in the unregularised construction.  No fitted value is accepted if it
is nonfinite or its constant moments are lost.
"""
function _local_joint_soft_row!(
        output::Vector{T}, baseline::Vector{T}, moments::Matrix{T},
        target::Vector{T}, pressure::Matrix{T}, pressure_target::Vector{T},
        commutator::Matrix{T}, scratch::MimeticJointFitScratch{T};
        pressure_weight::T, commutator_weight::T, correction_cap::T,
    ) where {T}
    if isempty(baseline)
        copyto!(output, baseline)
        return false
    end
    kinetic = scratch.ones_kinetic
    resize!(kinetic, length(baseline))
    fill!(kinetic, one(T))
    consistent = _local_commuting_fit(
        baseline, moments, target, kinetic;
        rank_tolerance = sqrt(eps(T)), amplification_limit = T(1.0e6),
    )
    if !consistent.success
        copyto!(output, baseline)
        return false
    end
    origin = consistent.values
    basis = nullspace(moments; atol = T(1.0e-10))
    if isempty(basis)
        copyto!(output, origin)
        return true
    end

    observation = scratch.observation
    coefficients = scratch.observation_coefficients
    right_hand_sides = scratch.observation_rhs
    empty!(coefficients)
    empty!(right_hand_sides)
    for (operator, desired, weight) in (
            (pressure, pressure_target, pressure_weight),
            (commutator, nothing, commutator_weight),
        )
        weight <= zero(T) && continue
        resize!(observation, size(operator, 2))
        for row in axes(operator, 1)
            for column in axes(operator, 2)
                observation[column] = operator[row, column]
            end
            scale = norm(observation)
            desired_value = isnothing(desired) ? zero(T) : desired[row]
            if scale <= T(64) * eps(T)
                if !(abs(desired_value) <= T(64) * eps(T))
                    copyto!(output, baseline)
                    return false
                end
                continue
            end
            factor = sqrt(weight) / scale
            projected = transpose(observation) * basis
            for value in projected
                push!(coefficients, factor * value)
            end
            push!(
                right_hand_sides,
                factor * (desired_value - dot(observation, origin))
            )
        end
    end
    observation_count = length(right_hand_sides)
    basis_count = size(basis, 2)
    dimensions = (observation_count + basis_count, basis_count)
    observations = get!(scratch.fit_matrices, dimensions) do
        Matrix{T}(undef, dimensions...)
    end
    ridge = T(0.1)
    ridge_scale = sqrt(ridge)
    for column in 1:basis_count
        for row in 1:observation_count
            observations[row, column] =
                coefficients[(row - 1) * basis_count + column]
        end
        for row in 1:basis_count
            observations[observation_count + row, column] =
                ridge_scale * (row == column ? one(T) : zero(T))
        end
    end
    fit_rhs = scratch.fit_rhs
    resize!(fit_rhs, observation_count + basis_count)
    copyto!(fit_rhs, 1, right_hand_sides, 1, observation_count)
    for row in 1:basis_count
        fit_rhs[observation_count + row] = zero(T)
    end
    # Rectangular `A \ b` uses pivoted QR on a copy of A. This buffer is
    # scratch and is fully overwritten on the next row, so perform the same
    # factorization in place. Preserve the square-system dispatch when no
    # observations survive, where `\` selects its diagonal fast path.
    fitted = observation_count == 0 ? observations \ fit_rhs :
        qr!(observations, ColumnNorm()) \ fit_rhs
    direction = basis * fitted
    if !all(isfinite, direction)
        copyto!(output, baseline)
        return false
    end
    fraction = min(
        one(T), correction_cap * norm(baseline) /
            max(norm(direction), eps(T))
    )
    result = origin + fraction * direction
    moment_error = norm(moments * result - target, Inf)
    success = all(isfinite, result) &&
        moment_error <= T(1.0e-8) * max(norm(target, Inf), one(T))
    copyto!(output, success ? result : baseline)
    return success
end

"""Rows of `reference*inv(K)` that overlap one local face patch."""
function _local_joint_overlap!(
        scratch::MimeticJointFitScratch{T}, reference, support, kinetic,
    ) where {T}
    overlapping = scratch.overlapping
    empty!(overlapping)
    for column in support
        append!(overlapping, scratch.matrix_rows_at_column[column])
    end
    sort!(unique!(overlapping))
    lookup = scratch.overlap_lookup
    for (position, row) in pairs(overlapping)
        lookup[row] = position
    end
    dimensions = (length(overlapping), length(support))
    values = get!(scratch.overlap_buffers, dimensions) do
        Matrix{T}(undef, dimensions...)
    end
    fill!(values, zero(T))
    matrix_rows = rowvals(reference)
    matrix_values = nonzeros(reference)
    for (position, column) in pairs(support)
        for pointer in nzrange(reference, column)
            values[lookup[matrix_rows[pointer]], position] =
                matrix_values[pointer]
        end
    end
    inverse_kinetic = if eltype(kinetic) === T
        buffer = scratch.inverse_kinetic
        resize!(buffer, length(support))
        for (position, column) in pairs(support)
            buffer[position] = inv(kinetic[column])
        end
        buffer
    else
        inv.(kinetic[support])
    end
    values .*= transpose(inverse_kinetic)
    return values
end

function _local_joint_constraint_operator(
        solver::MimeticProjectionSolver{T}, data, active_faces,
        relative_diagonal,
        scratch::MimeticJointFitScratch{T} = MimeticJointFitScratch(T),
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    reconstruction = _local_joint_reconstruction(
        solver, data, active_faces, relative_diagonal,
    )
    row_count, face_count = size(reconstruction.normal)
    row_count == 0 && return (;
        constraint = reconstruction.normal,
        candidate_constraint = reconstruction.normal,
        candidate_locations = reconstruction.candidate_cells,
        constraint_locations = reconstruction.candidate_cells,
        constraint_scales = reconstruction.candidate_scales,
        candidate_scales = reconstruction.candidate_scales,
        candidate_count = 0, dependent_count = 0,
        underresolved = reconstruction.underresolved,
        commuting_count = 0, fallback_count = 0,
        commuting_residual = zero(T),
        pressure_commuting_count = 0, pressure_fallback_count = 0,
        pressure_commuting_residual = zero(T),
        tangent = reconstruction.tangent,
    )

    # Grow patches through shared normal or tangential face support for
    # both cell-centred and face-centred row layouts.
    _local_joint_prepare!(scratch, row_count, face_count)
    _local_joint_graph!(
        scratch, reconstruction.normal, reconstruction.tangent, row_count,
    )
    references = _cell_references(mesh)
    centres = [
        cellcenter(
            mesh, reference.block, reference.i,
            reference.j
        ) for reference in references
    ]
    fractions = T[
        _conditional_fraction(interface, reference)
            for reference in references
    ]
    constant_samples = scratch.constant_samples
    common_pressure = scratch.common_pressure
    for (local_face, global_face) in pairs(active_faces)
        face = mesh.faces[global_face]
        constant_samples[local_face, face.axis] = one(T)
        displacement = _face_displacement(mesh, face)
        factor = data.coupling[global_face] * T(face.area) /
            relative_diagonal[local_face]
        common_pressure[local_face, 1] = factor * displacement[1]
        common_pressure[local_face, 2] = factor * displacement[2]
    end
    normal_moments = reconstruction.normal * constant_samples
    tangent_moments = _local_joint_tangent_moments(
        solver, reconstruction, normal_moments, constant_samples,
    )
    inverse_density_jump = inv(solver.density[2]) -
        inv(solver.density[1])

    for row in 1:row_count
        patch = scratch.rows[row]
        support = _local_joint_patch_support!(
            scratch, row, solver.patch_radius,
        )
        location = reconstruction.candidate_cells[row]
        centre, spacing = _local_joint_row_geometry(
            solver, mesh, references, centres, location,
        )
        eta = scratch.moment_target
        eta[1] = normal_moments[row, 1]
        eta[2] = normal_moments[row, 2]
        measure = hypot(eta...)
        unit_normal = scratch.unit_normal
        unit_normal .= eta ./ measure
        support_count = length(support)
        if size(patch.moments, 2) != support_count
            patch.moments = Matrix{T}(undef, 2, support_count)
            patch.pressure = Matrix{T}(undef, 4, support_count)
        end
        moment = patch.moments
        observations = patch.pressure
        for (position, local_face) in pairs(support)
            for axis in 1:2
                moment[axis, position] = constant_samples[local_face, axis]
                observations[axis, position] = common_pressure[local_face, axis]
            end
        end
        fitted_t, fitted_n = patch.base_t, patch.base_n
        resize!(fitted_t, support_count)
        resize!(fitted_n, support_count)
        fill!(fitted_t, zero(T))
        fill!(fitted_n, zero(T))
        for axis in 1:2
            positions = scratch.positions
            empty!(positions)
            for (position, local_face) in pairs(support)
                mesh.faces[active_faces[local_face]].axis == axis &&
                    push!(positions, position)
            end
            if isempty(positions)
                patch.geometry_fallback = true
                continue
            end
            weights = scratch.weights
            empty!(weights)
            for position in positions
                face = mesh.faces[active_faces[support[position]]]
                dx = _wrapped_delta(
                    face.center[1] - centre[1],
                    mesh.extent[1], mesh.periodic[1]
                ) / spacing[1]
                dy = _wrapped_delta(
                    face.center[2] - centre[2],
                    mesh.extent[2], mesh.periodic[2]
                ) / spacing[2]
                push!(weights, inv(dx^2 + dy^2 + T(0.25)^2))
            end
            weights ./= sum(weights)
            for (position, weight) in zip(positions, weights)
                fitted_t[position] = tangent_moments[row, axis] * weight
                fitted_n[position] = eta[axis] * weight
            end
        end
        resize!(patch.tangent_values, support_count)
        resize!(patch.normal_values, support_count)
        copyto!(patch.tangent_values, fitted_t)
        copyto!(patch.normal_values, fitted_n)

        for (position, local_face) in pairs(support)
            face = mesh.faces[active_faces[local_face]]
            minus = _cell_index(
                face.minus, mesh.block_size...,
            )
            plus = _cell_index(
                face.plus, mesh.block_size...,
            )
            local_coordinate(cell) = begin
                point = centres[cell]
                dx = _wrapped_delta(
                    point[1] - centre[1],
                    mesh.extent[1], mesh.periodic[1]
                )
                dy = _wrapped_delta(
                    point[2] - centre[2],
                    mesh.extent[2], mesh.periodic[2]
                )
                unit_normal[1] * dx + unit_normal[2] * dy
            end
            minus_distance = local_coordinate(minus)
            plus_distance = local_coordinate(plus)
            factor = data.coupling[active_faces[local_face]] *
                T(face.area) / relative_diagonal[local_face]
            observations[3, position] = factor * (
                fractions[plus] * plus_distance -
                    fractions[minus] * minus_distance
            )
            observations[4, position] = factor * (
                (one(T) - fractions[plus]) * plus_distance -
                    (one(T) - fractions[minus]) * minus_distance
            )
        end
        pressure_target = patch.pressure_target
        pressure_target[1] = inverse_density_jump * tangent_moments[row, 1]
        pressure_target[2] = inverse_density_jump * tangent_moments[row, 2]
        pressure_target[3] = zero(T)
        pressure_target[4] = zero(T)
    end

    for _ in 1:solver.fit_sweeps
        normal_matrix = _local_joint_matrix(
            scratch, row_count, face_count, false,
        )
        _local_joint_matrix_rows_at_columns!(scratch, normal_matrix)
        for row in 1:row_count
            patch = scratch.rows[row]
            overlap = _local_joint_overlap!(
                scratch, normal_matrix, patch.support, relative_diagonal,
            )
            target = scratch.moment_target
            target[1] = tangent_moments[row, 1]
            target[2] = tangent_moments[row, 2]
            patch.tangent_success = _local_joint_soft_row!(
                patch.tangent_values, patch.base_t, patch.moments,
                target, patch.pressure, patch.pressure_target, overlap, scratch;
                pressure_weight = solver.pressure_weight,
                commutator_weight = solver.commutator_weight,
                correction_cap = solver.correction_cap,
            )
        end
        tangent_matrix = _local_joint_matrix(
            scratch, row_count, face_count, true,
        )
        _local_joint_matrix_rows_at_columns!(scratch, tangent_matrix)
        for row in 1:row_count
            patch = scratch.rows[row]
            overlap = _local_joint_overlap!(
                scratch, tangent_matrix, patch.support, relative_diagonal,
            )
            target = scratch.moment_target
            target[1] = normal_moments[row, 1]
            target[2] = normal_moments[row, 2]
            patch.normal_success = _local_joint_soft_row!(
                patch.normal_values, patch.base_n, patch.moments,
                target, scratch.empty_pressure, scratch.empty_target,
                overlap, scratch;
                pressure_weight = zero(T),
                commutator_weight = solver.commutator_weight,
                correction_cap = solver.correction_cap,
            )
        end
    end
    tangent = _local_joint_matrix(scratch, row_count, face_count, true)
    candidate = _local_joint_matrix(scratch, row_count, face_count, false)
    inverse_kinetic = spdiagm(0 => inv.(relative_diagonal))
    product = tangent * inverse_kinetic * transpose(candidate)
    tangent_norms = sqrt.(
        diag(
            tangent * inverse_kinetic * transpose(tangent),
        )
    )
    normal_norms = sqrt.(
        diag(
            candidate * inverse_kinetic * transpose(candidate),
        )
    )
    commutator = maximum(
        (
            abs(product[row, column]) /
                max(tangent_norms[row] * normal_norms[column], sqrt(eps(T)))
                for row in axes(product, 1), column in axes(product, 2)
        ); init = zero(T)
    )
    inverse_density_scale = max(
        inv(solver.density[1]),
        inv(solver.density[2])
    )
    pressure_errors = scratch.pressure_errors
    empty!(pressure_errors)
    pressure_compatible = 0
    fallback_count = 0
    geometry_fallback_count = 0
    for row in 1:row_count
        patch = scratch.rows[row]
        observed = patch.pressure * patch.tangent_values
        scale = max(
            reconstruction.candidate_scales[row] *
                inverse_density_scale,
            sqrt(eps(T)),
        )
        error = norm(observed - patch.pressure_target, Inf) / scale
        push!(pressure_errors, error)
        location = reconstruction.candidate_cells[row]
        block = _local_joint_row_block(solver, mesh, references, location)
        local_h = max(mesh.blocks[block].spacing...) /
            max(mesh.extent...)
        pressure_compatible += error <= local_h
        fallback_count += !(patch.tangent_success & patch.normal_success)
        geometry_fallback_count += patch.geometry_fallback
    end
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    fallback_count += geometry_fallback_count
    return (;
        constraint = candidate[retained, :],
        candidate_constraint = candidate,
        candidate_locations = reconstruction.candidate_cells,
        constraint_locations = reconstruction.candidate_cells[retained],
        constraint_scales = reconstruction.candidate_scales[retained],
        candidate_scales = reconstruction.candidate_scales,
        candidate_count = row_count,
        dependent_count = row_count - length(retained),
        underresolved = reconstruction.underresolved +
            geometry_fallback_count,
        commuting_count = row_count - fallback_count,
        fallback_count, commuting_residual = commutator,
        pressure_commuting_count = pressure_compatible,
        pressure_fallback_count = row_count - pressure_compatible,
        pressure_commuting_residual = maximum(pressure_errors),
        tangent,
    )
end
