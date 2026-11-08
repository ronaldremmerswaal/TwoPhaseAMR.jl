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

function _local_joint_matrix(supports, values, face_count, ::Type{T}) where {T}
    rows = Int[]
    columns = Int[]
    coefficients = T[]
    for row in eachindex(supports)
        for (position, column) in pairs(supports[row])
            value = values[row][position]
            abs(value) <= T(64) * eps(T) && continue
            push!(rows, row)
            push!(columns, column)
            push!(coefficients, value)
        end
    end
    return sparse(rows, columns, coefficients,
        length(supports), face_count)
end

"""Regularised local fit in the exact constant-moment nullspace.

Each pressure/commutator observation is divided by its row norm before
weighting.  The cap is relative to the geometric row, which prevents the
exact pressure moment equations from creating the very large coefficients
seen in the unregularised construction.  No fitted value is accepted if it
is nonfinite or its constant moments are lost.
"""
function _local_joint_soft_row(
        baseline::Vector{T}, moments::Matrix{T}, target::Vector{T},
        pressure::Matrix{T}, pressure_target::Vector{T},
        commutator::Matrix{T}; pressure_weight::T,
        commutator_weight::T, correction_cap::T,
    ) where {T}
    length(baseline) == 0 && return baseline, false
    consistent = _local_commuting_fit(
        baseline, moments, target, ones(T, length(baseline));
        rank_tolerance = sqrt(eps(T)), amplification_limit = T(1e6),
    )
    consistent.success || return baseline, false
    origin = consistent.values
    basis = nullspace(moments; atol = T(1e-10))
    isempty(basis) && return origin, true

    observations = Matrix{T}[]
    right_hand_sides = Vector{T}[]
    for (operator, desired, weight) in (
            (pressure, pressure_target, pressure_weight),
            (commutator, zeros(T, size(commutator, 1)),
                commutator_weight),
        )
        weight <= zero(T) && continue
        for row in axes(operator, 1)
            observation = vec(operator[row, :])
            scale = norm(observation)
            if scale <= T(64) * eps(T)
                abs(desired[row]) <= T(64) * eps(T) ||
                    return baseline, false
                continue
            end
            factor = sqrt(weight) / scale
            push!(observations, reshape(
                factor .* (transpose(observation) * basis),
                1, size(basis, 2),
            ))
            push!(right_hand_sides, T[
                factor * (desired[row] - dot(observation, origin)),
            ])
        end
    end
    ridge = T(0.1)
    push!(observations, sqrt(ridge) .* Matrix{T}(
        I, size(basis, 2), size(basis, 2),
    ))
    push!(right_hand_sides, zeros(T, size(basis, 2)))
    direction = basis * (vcat(observations...) \
        vcat(right_hand_sides...))
    all(isfinite, direction) || return baseline, false
    fraction = min(one(T), correction_cap * norm(baseline) /
        max(norm(direction), eps(T)))
    result = origin + fraction * direction
    moment_error = norm(moments * result - target, Inf)
    success = all(isfinite, result) &&
        moment_error <= T(1e-8) * max(norm(target, Inf), one(T))
    return success ? result : baseline, success
end

"""Rows of `reference*inv(K)` that overlap one local face patch."""
function _local_joint_overlap(reference, rows_at_column, support, kinetic)
    overlapping = Int[]
    for column in support
        append!(overlapping, rows_at_column[column])
    end
    sort!(unique!(overlapping))
    values = Matrix(reference[overlapping, support])
    values .*= transpose(inv.(kinetic[support]))
    return values
end

function _local_joint_constraint_operator(
        solver::MimeticProjectionSolver{T}, data, active_faces,
        relative_diagonal,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    face_layout = solver.variant === :joint_face
    reconstruction = if face_layout
        # CMDGFM's beta only chooses *where* normal continuity is useful.
        # The fitted row below remains a physical normal observation and
        # enters the same symmetric KKT system as the cell-row variant.
        face_rows = _local_joint_face_rows(
            solver, data, active_faces, relative_diagonal,
        )
        (;
            normal = face_rows.candidate_constraint,
            tangent = spzeros(T, face_rows.candidate_count,
                length(active_faces)),
            candidate_cells = face_rows.candidate_locations,
            candidate_scales = face_rows.candidate_scales,
            underresolved = face_rows.underresolved,
        )
    else
        _local_aperture_normal_tangent_reconstruction(
            solver, data, active_faces,
        )
    end
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

    # An interface row starts from its aperture Gauss vector.  The patch is
    # formed from neighboring mixed-cell rows, so there are no PLIC nodes or
    # independent facewise normal/tangent labels.
    normal_supports = _local_sparse_row_supports(reconstruction.normal)
    tangent_supports = _local_sparse_row_supports(reconstruction.tangent)
    graph_supports = [sort!(unique!(vcat(
        normal_supports[row], tangent_supports[row],
    ))) for row in 1:row_count]
    graph_rows_at_column = _local_rows_at_columns(
        graph_supports, face_count,
    )
    references = _cell_references(mesh)
    centres = [cellcenter(mesh, reference.block, reference.i,
        reference.j) for reference in references]
    fractions = T[_conditional_fraction(interface, reference)
        for reference in references]
    constant_samples = zeros(T, face_count, 2)
    common_pressure = zeros(T, face_count, 2)
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
    tangent_moments = if face_layout
        hcat(-normal_moments[:, 2], normal_moments[:, 1])
    else
        reconstruction.tangent * constant_samples
    end
    inverse_density_jump = inv(solver.density[2]) -
        inv(solver.density[1])

    supports = Vector{Vector{Int}}(undef, row_count)
    base_t = Vector{Vector{T}}(undef, row_count)
    base_n = Vector{Vector{T}}(undef, row_count)
    moments = Vector{Matrix{T}}(undef, row_count)
    pressure = Vector{Matrix{T}}(undef, row_count)
    pressure_targets = Vector{Vector{T}}(undef, row_count)
    geometry_fallback = falses(row_count)
    for row in 1:row_count
        patch_rows = _local_patch_rows(
            row, solver.patch_radius, graph_supports,
            graph_rows_at_column,
        )
        support = Int[]
        for patch_row in patch_rows
            append!(support, graph_supports[patch_row])
        end
        sort!(unique!(support))
        supports[row] = support
        location = reconstruction.candidate_cells[row]
        reference = face_layout ? nothing : references[location]
        face_location = face_layout ? mesh.faces[location] : nothing
        centre = face_layout ? face_location.center : centres[location]
        spacing = face_layout ?
            mesh.blocks[face_location.minus.block].spacing :
            mesh.blocks[reference.block].spacing
        eta = vec(normal_moments[row, :])
        measure = hypot(eta...)
        unit_normal = eta ./ measure
        moment = transpose(constant_samples[support, :])
        moments[row] = moment
        fitted_t = zeros(T, length(support))
        fitted_n = zeros(T, length(support))
        for axis in 1:2
            positions = [position for (position, local_face) in
                pairs(support) if mesh.faces[active_faces[local_face]].axis == axis]
            if isempty(positions)
                geometry_fallback[row] = true
                continue
            end
            weights = T[]
            for position in positions
                face = mesh.faces[active_faces[support[position]]]
                dx = _wrapped_delta(face.center[1] - centre[1],
                    mesh.extent[1], mesh.periodic[1]) / spacing[1]
                dy = _wrapped_delta(face.center[2] - centre[2],
                    mesh.extent[2], mesh.periodic[2]) / spacing[2]
                push!(weights, inv(dx^2 + dy^2 + T(0.25)^2))
            end
            weights ./= sum(weights)
            for (position, weight) in zip(positions, weights)
                fitted_t[position] = tangent_moments[row, axis] * weight
                fitted_n[position] = eta[axis] * weight
            end
        end
        base_t[row] = fitted_t
        base_n[row] = fitted_n

        observations = zeros(T, 4, length(support))
        observations[1:2, :] .= transpose(common_pressure[support, :])
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
                dx = _wrapped_delta(point[1] - centre[1],
                    mesh.extent[1], mesh.periodic[1])
                dy = _wrapped_delta(point[2] - centre[2],
                    mesh.extent[2], mesh.periodic[2])
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
        pressure[row] = observations
        pressure_targets[row] = T[
            inverse_density_jump * tangent_moments[row, 1],
            inverse_density_jump * tangent_moments[row, 2],
            zero(T), zero(T),
        ]
    end

    tangent_values = copy.(base_t)
    normal_values = copy.(base_n)
    tangent_success = trues(row_count)
    normal_success = trues(row_count)
    for _ in 1:solver.fit_sweeps
        normal_matrix = _local_joint_matrix(
            supports, normal_values, face_count, T,
        )
        normal_rows_at_column = _local_rows_at_columns(
            _local_sparse_row_supports(normal_matrix), face_count,
        )
        for row in 1:row_count
            overlap = _local_joint_overlap(
                normal_matrix, normal_rows_at_column,
                supports[row], relative_diagonal,
            )
            tangent_values[row], tangent_success[row] =
                _local_joint_soft_row(
                    base_t[row], moments[row],
                    vec(tangent_moments[row, :]), pressure[row],
                    pressure_targets[row], overlap;
                    pressure_weight = solver.pressure_weight,
                    commutator_weight =
                        solver.commutator_weight,
                    correction_cap = solver.correction_cap,
                )
        end
        tangent_matrix = _local_joint_matrix(
            supports, tangent_values, face_count, T,
        )
        tangent_rows_at_column = _local_rows_at_columns(
            _local_sparse_row_supports(tangent_matrix), face_count,
        )
        for row in 1:row_count
            overlap = _local_joint_overlap(
                tangent_matrix, tangent_rows_at_column,
                supports[row], relative_diagonal,
            )
            normal_values[row], normal_success[row] =
                _local_joint_soft_row(
                    base_n[row], moments[row],
                    vec(normal_moments[row, :]),
                    zeros(T, 0, length(supports[row])), T[], overlap;
                    pressure_weight = zero(T),
                    commutator_weight =
                        solver.commutator_weight,
                    correction_cap = solver.correction_cap,
                )
        end
    end
    tangent = _local_joint_matrix(
        supports, tangent_values, face_count, T,
    )
    candidate = _local_joint_matrix(
        supports, normal_values, face_count, T,
    )
    inverse_kinetic = spdiagm(0 => inv.(relative_diagonal))
    product = tangent * inverse_kinetic * transpose(candidate)
    tangent_norms = sqrt.(diag(
        tangent * inverse_kinetic * transpose(tangent),
    ))
    normal_norms = sqrt.(diag(
        candidate * inverse_kinetic * transpose(candidate),
    ))
    commutator = maximum((
        abs(product[row, column]) /
            max(tangent_norms[row] * normal_norms[column], sqrt(eps(T)))
        for row in axes(product, 1), column in axes(product, 2)
    ); init = zero(T))
    inverse_density_scale = max(inv(solver.density[1]),
        inv(solver.density[2]))
    pressure_errors = T[]
    pressure_compatible = 0
    for row in 1:row_count
        observed = pressure[row] * tangent_values[row]
        scale = max(
            reconstruction.candidate_scales[row] *
                inverse_density_scale,
            sqrt(eps(T)),
        )
        error = norm(observed - pressure_targets[row], Inf) / scale
        push!(pressure_errors, error)
        location = reconstruction.candidate_cells[row]
        block = face_layout ? mesh.faces[location].minus.block :
            references[location].block
        local_h = max(mesh.blocks[block].spacing...) /
            max(mesh.extent...)
        pressure_compatible += error <= local_h
    end
    retained = _local_independent_rows(
        candidate, relative_diagonal, solver.rank_tolerance, T,
    )
    fallback_count = count(!, tangent_success .& normal_success) +
        count(geometry_fallback)
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
            count(geometry_fallback),
        commuting_count = row_count - fallback_count,
        fallback_count, commuting_residual = commutator,
        pressure_commuting_count = pressure_compatible,
        pressure_fallback_count = row_count - pressure_compatible,
        pressure_commuting_residual = maximum(pressure_errors),
        tangent,
    )
end
