module CoDesignNT

using LinearAlgebra
using Printf
using SparseArrays
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "vortex_sheet_commutativity",
    "commuting_normal_interpolant.jl"))
using .VortexSheetCommutativity: build_problem

"""A cheap local test of physical N and T, with no pressure/KKT solve.

Rows act on the production active face-jump space. The circle normal and
tangent are analytical so geometry error does not conceal algebraic error.
`K` and the pressure-only jump are the production face metric and coupling.
All methods use the same sparse 2.5-cell patches from the earlier focused
commutativity experiment. Pressure modes 1:4 are separate x/y sines and
cosines; diagonal modes 5:6 are held out. Modes 7:8 are signed-distance
pressure in just one phase, with zero tangential trace derivative. No result
here changes a production operator.
"""

const TWOPI = 2pi

function pressure_mode(j, x, y)
    j == 1 && return sin(TWOPI * x)
    j == 2 && return sin(TWOPI * y)
    j == 3 && return cos(TWOPI * x)
    j == 4 && return cos(TWOPI * y)
    j == 5 && return sin(TWOPI * (x + y))
    j == 6 && return sin(TWOPI * (x - y))
    error("unknown pressure mode")
end

function pressure_gradient(j, x, y)
    j == 1 && return (TWOPI * cos(TWOPI * x), 0.0)
    j == 2 && return (0.0, TWOPI * cos(TWOPI * y))
    j == 3 && return (-TWOPI * sin(TWOPI * x), 0.0)
    j == 4 && return (0.0, -TWOPI * sin(TWOPI * y))
    j == 5 && return (TWOPI * cos(TWOPI * (x + y)),
        TWOPI * cos(TWOPI * (x + y)))
    j == 6 && return (TWOPI * cos(TWOPI * (x - y)),
        -TWOPI * cos(TWOPI * (x - y)))
    error("unknown pressure mode")
end

function velocity_mode(j, x, y)
    j == 1 && return (
        sin(TWOPI * x) + 0.2cos(TWOPI * y),
        cos(TWOPI * y) + 0.1sin(TWOPI * x),
    )
    j == 2 && return (
        cos(TWOPI * (x + y)), sin(TWOPI * (x - y)),
    )
    error("unknown velocity mode")
end

dot2(a, b) = a[1] * b[1] + a[2] * b[2]

function samples(problem)
    mesh = problem.mesh
    active = problem.active_faces
    assembly = problem.assembly
    k = assembly.relative_diagonal
    references = TwoPhaseAMR._cell_references(mesh)
    xp = zeros(length(active), 8)
    sp = zeros(length(problem.row_points), 8)
    xa = zeros(length(active), 2)
    sa = zeros(length(problem.row_points), 2)
    alpha = inv(problem.density[2]) - inv(problem.density[1])
    for j in axes(xp, 2)
        p = [begin
            x, y = cellcenter(mesh, reference.block, reference.i,
                reference.j)
            if j <= 6
                pressure_mode(j, x, y)
            else
                distance = hypot(x - problem.centre[1],
                    y - problem.centre[2]) - problem.radius
                belongs = j == 7 ? distance < 0 : distance > 0
                belongs ? distance : 0.0
            end
        end for reference in references]
        xp[:, j] .= assembly.data.coupling[active] .*
            (assembly.incidence * p)[active] ./ k
        for row in eachindex(problem.row_points)
            point = problem.row_points[row]
            sp[row, j] = j <= 6 ?
                alpha * dot2(problem.row_tangents[row],
                    pressure_gradient(j, point...)) : 0.0
        end
    end
    # Local affine pressure modes are the polynomial patch test for the
    # pressure commutator. The circle is away from periodic seams, so the
    # periodic incidence sees ordinary affine differences on every patch.
    for axis in 1:2
        p = [cellcenter(mesh, reference.block, reference.i,
            reference.j)[axis] for reference in references]
        xa[:, axis] .= assembly.data.coupling[active] .*
            (assembly.incidence * p)[active] ./ k
        for row in eachindex(problem.row_points)
            sa[row, axis] = alpha * problem.row_tangents[row][axis]
        end
    end
    zv = zeros(length(active), 2)
    tv = zeros(length(problem.row_points), 2)
    nv = similar(tv)
    for (local_face, face_index) in pairs(active)
        face = mesh.faces[face_index]
        for j in 1:2
            zv[local_face, j] = velocity_mode(j, face.center...)[face.axis]
        end
    end
    for row in eachindex(problem.row_points), j in 1:2
        v = velocity_mode(j, problem.row_points[row]...)
        tv[row, j] = dot2(problem.row_tangents[row], v)
        nv[row, j] = dot2(problem.row_normals[row], v)
    end
    return (; xp, sp, xa, sa, zv, tv, nv)
end

function local_support(problem, row)
    ti, _ = findnz(problem.tangent[row, :])
    ni, _ = findnz(problem.normal_baseline[row, :])
    indices = sort!(unique(vcat(ti, ni)))
    t = Vector(problem.tangent[row, indices])
    n = Vector(problem.normal_baseline[row, indices])
    return indices, t, n
end

function constant_moments(problem, indices)
    result = zeros(2, length(indices))
    for (local_face, index) in pairs(indices)
        axis = problem.mesh.faces[problem.active_faces[index]].axis
        result[axis, local_face] = 1
    end
    return result
end

"""All nearby rows of a fixed operator, tested against one local row.

This is the complete matrix commutator restricted to the chosen physical
face support, not merely the matching-row inner product.
"""
function overlap_constraints(reference, indices, k)
    rows = Matrix(reference[:, indices]) .* transpose(inv.(k))
    retained = [row for row in axes(rows, 1)
        if norm(view(rows, row, :)) > 1e-14]
    return rows[retained, :]
end

"""Least-change solution of small linear moment constraints.

Rows are normalized before an SVD, so pressure modes and velocity moments
are compared by relative row scale. A large residual signals an infeasible
or numerically rank-deficient local problem, rather than silently accepting
a spurious commuting identity.
"""
function enforce(base, constraints, target; tolerance = 1e-10)
    a = Matrix{Float64}(constraints)
    b = Vector{Float64}(target)
    for row in axes(a, 1)
        scale = norm(view(a, row, :))
        scale > 1e-14 || return base, Inf
        @views a[row, :] ./= scale
        b[row] /= scale
    end
    factor = svd(a; full = false)
    largest = maximum(factor.S; init = 0.0)
    rank_a = count(>(tolerance * largest), factor.S)
    if rank_a == 0
        return base, maximum(abs, b)
    end
    residual = b - a * base
    correction = factor.V[:, 1:rank_a] *
        ((transpose(factor.U[:, 1:rank_a]) * residual) ./
            factor.S[1:rank_a])
    result = base + correction
    return result, maximum(abs, a * result - b)
end

"""A pressure correction confined to the exact moment/commutator nullspace.

Unlike sequentially fitting all rows with a truncated pseudoinverse, this
cannot undo the normal/tangent moment constraints when pressure moments are
nearly dependent on them. Small pressure singular values are discarded;
the caller separately caps the size of the resulting direction.
"""
function nullspace_pressure_direction(base, constraints, pressure, target;
        moment_tolerance = 1e-10, pressure_tolerance = 1e-3)
    a = Matrix{Float64}(constraints)
    for row in axes(a, 1)
        scale = norm(view(a, row, :))
        scale > 1e-14 || return zeros(length(base))
        @views a[row, :] ./= scale
    end
    fa = svd(a; full = true)
    rank_a = count(>(moment_tolerance * maximum(fa.S)), fa.S)
    rank_a < length(base) || return zeros(length(base))
    z = fa.V[:, (rank_a + 1):end]
    p = Matrix{Float64}(pressure)
    b = Vector{Float64}(target) - p * base
    for row in axes(p, 1)
        scale = norm(view(p, row, :))
        scale > 1e-14 || continue
        @views p[row, :] ./= scale
        b[row] /= scale
    end
    projected = p * z
    fp = svd(projected; full = false)
    isempty(fp.S) && return zeros(length(base))
    rank_p = count(>(pressure_tolerance * maximum(fp.S)), fp.S)
    rank_p == 0 && return zeros(length(base))
    coefficients = fp.V[:, 1:rank_p] *
        ((transpose(fp.U[:, 1:rank_p]) * b) ./ fp.S[1:rank_p])
    return z * coefficients
end

"""Bounded soft pressure/commutator fit with exact constant moments."""
function soft_row(base, moments, target, pressure, pressure_target,
        commutator; pressure_weight = 10.0, commutator_weight = 1.0,
        ridge = 0.1, correction_cap = 1.0)
    base_exact, = enforce(base, moments, target)
    a = Matrix{Float64}(moments)
    for row in axes(a, 1)
        scale = norm(view(a, row, :))
        @views a[row, :] ./= scale
    end
    fa = svd(a; full = true)
    rank_a = count(>(1e-10 * maximum(fa.S)), fa.S)
    rank_a < length(base) || return base_exact
    z = fa.V[:, (rank_a + 1):end]
    blocks = Matrix{Float64}[]
    targets = Vector{Float64}[]
    for (operator, desired, weight) in (
            (pressure, pressure_target, pressure_weight),
            (commutator, zeros(size(commutator, 1)),
                commutator_weight))
        weight > 0 || continue
        for row in axes(operator, 1)
            observation = Vector(view(operator, row, :))
            scale = norm(observation)
            scale > 1e-14 || continue
            push!(blocks, reshape(
                sqrt(weight) .* (transpose(observation) * z) ./ scale,
                1, size(z, 2),
            ))
            push!(targets, [sqrt(weight) *
                (desired[row] - dot(observation, base_exact)) / scale])
        end
    end
    push!(blocks, sqrt(ridge) .* Matrix{Float64}(I, size(z, 2),
        size(z, 2)))
    push!(targets, zeros(size(z, 2)))
    coefficients = vcat(blocks...) \ vcat(targets...)
    direction = z * coefficients
    alpha = min(1.0, correction_cap * norm(base) /
        max(norm(direction), eps()))
    return base_exact + alpha * direction
end

function row_matrix(problem, trials, component)
    matrix = spzeros(Float64, length(trials), length(problem.active_faces))
    for (row, trial) in pairs(trials)
        values = getproperty(trial, component)
        for (position, face) in pairs(trial.indices)
            matrix[row, face] = values[position]
        end
    end
    return matrix
end

"""Alternating local fits; each update sees the full overlapping operator."""
function alternating_candidates(problem, data; commutator_weight = 1.0,
        pressure_weight = 10.0, correction_cap = 1.0,
        pressure_basis = :trigonometric, iterations = 3)
    originals = [local_support(problem, row)
        for row in eachindex(problem.row_points)]
    trials = [(
        indices = indices, t = t, n = n,
        k = problem.assembly.relative_diagonal[indices],
        residual = 0.0, amplification = 1.0,
    ) for (indices, t, n) in originals]
    for _ in 1:iterations
        n_matrix = row_matrix(problem, trials, :n)
        t_updated = NamedTuple[]
        for (row, trial) in pairs(trials)
            (; indices, k) = trial
            moments = constant_moments(problem, indices)
            pressure, pressure_target = if pressure_basis === :affine
                transpose(data.xa[indices, :]), data.sa[row, :]
            elseif pressure_basis === :affine_plus_normal
                transpose(hcat(data.xa[indices, :],
                    data.xp[indices, 7:8])),
                vcat(data.sa[row, :], zeros(2))
            elseif pressure_basis === :trigonometric
                transpose(data.xp[indices, 1:4]), data.sp[row, 1:4]
            else
                error("unknown pressure basis: $pressure_basis")
            end
            overlap = overlap_constraints(n_matrix, indices, k)
            tangent = collect(problem.row_tangents[row])
            t = soft_row(originals[row][2], moments, tangent,
                pressure, pressure_target, overlap;
                commutator_weight, pressure_weight, correction_cap)
            push!(t_updated, merge(trial, (; t)))
        end
        t_matrix = row_matrix(problem, t_updated, :t)
        n_updated = NamedTuple[]
        for (row, trial) in pairs(t_updated)
            (; indices, k) = trial
            moments = constant_moments(problem, indices)
            overlap = overlap_constraints(t_matrix, indices, k)
            normal = collect(problem.row_normals[row])
            n = soft_row(originals[row][3], moments, normal,
                zeros(0, length(indices)), zeros(0), overlap;
                pressure_weight = 0.0, commutator_weight,
                correction_cap)
            baseline_t = originals[row][2]
            baseline_n = originals[row][3]
            amplification = max(norm(trial.t) / norm(baseline_t),
                norm(n) / norm(baseline_n))
            push!(n_updated, merge(trial, (; n, amplification)))
        end
        trials = n_updated
    end
    return trials
end

function two_face_rows(problem, indices, point; split = false)
    t = zeros(length(indices))
    n = zeros(length(indices))
    normal = problem.row_normals[point.row]
    tangent = problem.row_tangents[point.row]
    for axis in 1:2
        candidates = [local_face for (local_face, index) in pairs(indices)
            if problem.mesh.faces[problem.active_faces[index]].axis == axis]
        length(candidates) >= (split ? 2 : 1) || return nothing
        sort!(candidates; by = local_face -> begin
            face = problem.mesh.faces[problem.active_faces[indices[local_face]]]
            hypot(face.center[1] - point.x, face.center[2] - point.y)
        end)
        t[first(candidates)] = tangent[axis]
        n[candidates[split ? 2 : 1]] = normal[axis]
    end
    return t, n
end

"""Two globally disjoint interleaved face sets, both locally two-dimensional.

The checkerboard color is fixed once for the entire face mesh. All T rows
touch only color 0 and all N rows touch only color 1, so the *full* matrix
product T*inv(K)*N' vanishes without a dense projection. Each axis/color
uses normalized local inverse-distance weights and reproduces constants.
"""
function partition_rows(problem, indices, row; both_apertures = false)
    mesh = problem.mesh
    h = mesh.blocks[1].spacing[1]
    point = problem.row_points[row]
    tangent = problem.row_tangents[row]
    normal = problem.row_normals[row]
    t = zeros(length(indices))
    n = zeros(length(indices))
    for axis in 1:2, color in 0:1
        candidates = Int[]
        weights = Float64[]
        for (position, face_index) in pairs(indices)
            global_face = problem.active_faces[face_index]
            face = mesh.faces[global_face]
            face.axis == axis || continue
            if both_apertures
                data = problem.assembly.data
                min(data.liquid_aperture[global_face],
                    data.gas_aperture[global_face]) >
                        problem.interface.fraction_floor || continue
            end
            face_color = mod(
                floor(Int, face.center[1] / h + 1e-9) +
                floor(Int, face.center[2] / h + 1e-9), 2,
            )
            face_color == color || continue
            distance2 = abs2(face.center[1] - point[1]) +
                abs2(face.center[2] - point[2])
            push!(candidates, position)
            push!(weights, inv(distance2 + abs2(0.25h)))
        end
        isempty(candidates) && return nothing
        weights ./= sum(weights)
        for (position, weight) in zip(candidates, weights)
            if color == 0
                t[position] = tangent[axis] * weight
            else
                n[position] = normal[axis] * weight
            end
        end
    end
    return t, n
end

function candidate(problem, data, row, method)
    indices, t0, n0 = local_support(problem, row)
    k = problem.assembly.relative_diagonal[indices]
    c = constant_moments(problem, indices)
    tangent = collect(problem.row_tangents[row])
    normal = collect(problem.row_normals[row])
    residual = 0.0
    if method === :geometric
        t, n = t0, n0
    elseif method === :two_face || method === :split_support
        point = problem.row_points[row]
        result = two_face_rows(problem, indices,
            (; row, x = point[1], y = point[2]);
            split = method === :split_support)
        isnothing(result) && return nothing
        t, n = result
    elseif method in (:partition, :partition_pressure4_bounded,
            :partition_open, :partition_open_pressure4_bounded,
            :partition_open_affine_bounded)
        both_apertures = method in (:partition_open,
            :partition_open_pressure4_bounded,
            :partition_open_affine_bounded)
        result = partition_rows(problem, indices, row; both_apertures)
        isnothing(result) && return nothing
        t, n = result
        if method in (:partition_pressure4_bounded,
                :partition_open_pressure4_bounded,
                :partition_open_affine_bounded)
            tangent_positions = findall(!iszero, t)
            base = t[tangent_positions]
            moments = c[:, tangent_positions]
            affine = method === :partition_open_affine_bounded
            pressure = affine ?
                transpose(data.xa[indices[tangent_positions], :]) :
                transpose(data.xp[indices[tangent_positions], 1:4])
            target_pressure = affine ? data.sa[row, :] :
                data.sp[row, 1:4]
            direction = nullspace_pressure_direction(
                base, moments, pressure, target_pressure,
            )
            alpha = min(1.0, norm(base) / max(norm(direction), eps()))
            t[tangent_positions] .= base + alpha * direction
            residual = maximum(abs,
                moments * t[tangent_positions] - tangent)
        end
    elseif method === :N_from_T
        t = t0
        n, residual = enforce(n0,
            [c; transpose(t ./ k)], [normal; 0.0])
    elseif method === :N_full_from_T
        t = t0
        orth = overlap_constraints(problem.tangent, indices, k)
        n, residual = enforce(n0,
            [c; orth], [normal; zeros(size(orth, 1))])
    elseif method === :T_from_N
        n = n0
        t, residual = enforce(t0,
            [c; transpose(n ./ k)], [tangent; 0.0])
    elseif method === :T_full_from_N
        n = n0
        orth = overlap_constraints(problem.normal_baseline, indices, k)
        t, residual = enforce(t0,
            [c; orth], [tangent; zeros(size(orth, 1))])
    elseif method === :pressure2_T_then_N ||
            method === :pressure4_T_then_N
        # Fit T to local pressure moments, then build a K-orthogonal N.
        train = method === :pressure2_T_then_N ? (1:2) : (1:4)
        n = n0
        pressure = transpose(data.xp[indices, train])
        t, residual_t = enforce(t0,
            [c; pressure], [tangent; data.sp[row, train]])
        n, residual_n = enforce(n0,
            [c; transpose(t ./ k)], [normal; 0.0])
        residual = max(residual_t, residual_n)
    elseif method === :pressure4_bounded_T_from_N
        # Preserve normal consistency and exact K orthogonality. The full
        # pressure fit can be arbitrarily large, so cap the added correction
        # at the norm of the original physical tangent row.
        n = n0
        base_constraints = [c; transpose(n ./ k)]
        base_target = [tangent; 0.0]
        base, residual_base = enforce(
            t0, base_constraints, base_target,
        )
        pressure = transpose(data.xp[indices, 1:4])
        fitted, residual_fit = enforce(base,
            [base_constraints; pressure],
            [base_target; data.sp[row, 1:4]],
        )
        direction = fitted - base
        alpha = min(1.0, norm(t0) / max(norm(direction), eps()))
        t = base + alpha * direction
        residual = max(residual_base,
            maximum(abs, base_constraints * t - base_target))
        isfinite(residual_fit) || (residual = Inf)
    elseif method === :pressure4_bounded_T_from_N_full
        n = n0
        orth = overlap_constraints(problem.normal_baseline, indices, k)
        base_constraints = [c; orth]
        base_target = [tangent; zeros(size(orth, 1))]
        base, residual_base = enforce(
            t0, base_constraints, base_target,
        )
        pressure = transpose(data.xp[indices, 1:4])
        direction = nullspace_pressure_direction(
            base, base_constraints, pressure, data.sp[row, 1:4],
        )
        alpha = min(1.0, norm(t0) / max(norm(direction), eps()))
        t = base + alpha * direction
        residual = max(residual_base,
            maximum(abs, base_constraints * t - base_target))
    elseif method === :affine_bounded_T_from_N_full
        n = n0
        orth = overlap_constraints(problem.normal_baseline, indices, k)
        base_constraints = [c; orth]
        base_target = [tangent; zeros(size(orth, 1))]
        base, residual_base = enforce(
            t0, base_constraints, base_target,
        )
        pressure = transpose(data.xa[indices, :])
        direction = nullspace_pressure_direction(
            base, base_constraints, pressure, data.sa[row, :],
        )
        alpha = min(1.0, norm(t0) / max(norm(direction), eps()))
        t = base + alpha * direction
        residual = max(residual_base,
            maximum(abs, base_constraints * t - base_target))
    elseif method === :affine_exact_T_from_N ||
            method === :affine_bounded_T_from_N
        n = n0
        base_constraints = [c; transpose(n ./ k)]
        base_target = [tangent; 0.0]
        base, residual_base = enforce(
            t0, base_constraints, base_target,
        )
        pressure = transpose(data.xa[indices, :])
        fitted, residual_fit = enforce(base,
            [base_constraints; pressure],
            [base_target; data.sa[row, :]],
        )
        if method === :affine_bounded_T_from_N
            direction = fitted - base
            alpha = min(1.0, norm(t0) / max(norm(direction), eps()))
            t = base + alpha * direction
        else
            t = fitted
        end
        residual = max(residual_base,
            maximum(abs, base_constraints * t - base_target))
        isfinite(residual_fit) || (residual = Inf)
    else
        error("unknown candidate: $method")
    end
    return (; indices, t, n, k, residual,
        amplification = max(norm(t) / norm(t0), norm(n) / norm(n0)))
end

function study_case(n, density; reconstruction_radius = 2.5,
        soft_pressure_weight = 10.0, soft_cap = 1.0,
        soft_pressure_basis = :trigonometric,
        methods = (:geometric, :two_face, :split_support,
            :partition, :partition_pressure4_bounded,
            :partition_open, :partition_open_pressure4_bounded,
            :partition_open_affine_bounded,
            :N_from_T, :N_full_from_T, :T_from_N, :T_full_from_N,
            :pressure2_T_then_N,
            :pressure4_T_then_N, :pressure4_bounded_T_from_N,
            :pressure4_bounded_T_from_N_full,
            :affine_exact_T_from_N, :affine_bounded_T_from_N,
            :affine_bounded_T_from_N_full,
            :soft1, :soft10, :soft100))
    problem = build_problem(n; density, reconstruction_radius)
    data = samples(problem)
    results = NamedTuple[]
    for method in methods
        prepared = startswith(string(method), "soft") ?
            alternating_candidates(problem, data;
                commutator_weight = parse(Float64,
                    replace(string(method), "soft" => "")),
                pressure_weight = soft_pressure_weight,
                correction_cap = soft_cap,
                pressure_basis = soft_pressure_basis) : nothing
        sums = zeros(5)
        maxima = zeros(5)
        max_comm = max_amp = max_residual = 0.0
        support = rows = 0
        trials = NamedTuple[]
        angles = Float64[]
        for row in eachindex(problem.row_points)
            trial = isnothing(prepared) ?
                candidate(problem, data, row, method) : prepared[row]
            isnothing(trial) && continue
            (; indices, t, n, k) = trial
            rows += 1
            push!(trials, trial)
            point = problem.row_points[row]
            push!(angles, atan(point[2] - problem.centre[2],
                point[1] - problem.centre[1]))
            support += count(i -> abs(t[i]) > 1e-12 || abs(n[i]) > 1e-12,
                eachindex(indices))
            max_amp = max(max_amp, trial.amplification)
            max_residual = max(max_residual, trial.residual)
            comm = dot(t ./ k, n) /
                sqrt(dot(t ./ k, t) * dot(n ./ k, n))
            max_comm = max(max_comm, abs(comm))
            defects = (
                maximum(abs, transpose(t) * data.zv[indices, :] .-
                    transpose(data.tv[row, :])),
                maximum(abs, transpose(n) * data.zv[indices, :] .-
                    transpose(data.nv[row, :])),
                maximum(abs, transpose(t) * data.xp[indices, 1:4] .-
                    transpose(data.sp[row, 1:4])),
                maximum(abs, transpose(t) * data.xp[indices, 5:6] .-
                    transpose(data.sp[row, 5:6])),
                maximum(abs, transpose(t) * data.xp[indices, 7:8] .-
                    transpose(data.sp[row, 7:8])),
            )
            for j in 1:5
                sums[j] += abs2(defects[j])
                maxima[j] = max(maxima[j], defects[j])
            end
        end
        rows > 0 || error("no rows for $method at n=$n")
        # The matrix identity includes cross-row couplings. Same-row
        # orthogonality alone is insufficient when local patches overlap.
        tmat = spzeros(Float64, rows, length(problem.active_faces))
        nmat = spzeros(Float64, rows, length(problem.active_faces))
        for (row, trial) in pairs(trials)
            for (position, face) in pairs(trial.indices)
                tmat[row, face] = trial.t[position]
                nmat[row, face] = trial.n[position]
            end
        end
        inverse_k = spdiagm(0 => inv.(problem.assembly.relative_diagonal))
        global_commutator = tmat * inverse_k * transpose(nmat)
        tangent_norms = sqrt.(diag(tmat * inverse_k * transpose(tmat)))
        normal_norms = sqrt.(diag(nmat * inverse_k * transpose(nmat)))
        cross_row_comm = 0.0
        for column in 1:rows, row in 1:rows
            row == column && continue
            denominator = tangent_norms[row] * normal_norms[column]
            denominator > 0 || continue
            cross_row_comm = max(cross_row_comm,
                abs(global_commutator[row, column]) / denominator)
        end
        normalised_commutator = spdiagm(0 => inv.(tangent_norms)) *
            global_commutator * spdiagm(0 => inv.(normal_norms))
        smooth_comm = maximum(
            norm(normalised_commutator * mode) / norm(mode)
            for mode in (
                ones(rows), sin.(angles), cos.(angles),
                sin.(2 .* angles), cos.(2 .* angles),
            )
        )
        push!(results, (; n, density, method, rows,
            total_rows = length(problem.row_points),
            support = support / rows, max_amp, max_comm, cross_row_comm,
            smooth_comm,
            max_residual, l2 = sqrt.(sums ./ rows), linf = maxima))
    end
    return results
end

"""Report rank/conditioning of the metric-dual trial space E=K⁻¹[T' N'].

The same nonzero spectrum is obtained from the row Gram matrix
`[T;N] * inv(K) * [T;N]'`. This is only a rank diagnostic, not a
stability bound for a projection using the reduced space.
"""
function gram_diagnostics(n, density; methods = (:geometric,
        :N_full_from_T, :T_full_from_N, :partition, :soft100),
        soft_pressure_weight = 1000.0,
        soft_pressure_basis = :affine)
    problem = build_problem(n; density)
    data = samples(problem)
    inverse_k = spdiagm(0 => inv.(problem.assembly.relative_diagonal))
    println("method\trows\tT_rank\tT_cond\tN_rank\tN_cond\tcombined_rank\tcombined_cond\tactive_faces")
    for method in methods
        trials = if startswith(string(method), "soft")
            alternating_candidates(problem, data;
                commutator_weight = parse(Float64,
                    replace(string(method), "soft" => "")),
                pressure_weight = soft_pressure_weight,
                pressure_basis = soft_pressure_basis)
        else
            filter(!isnothing, [candidate(problem, data, row, method)
                for row in eachindex(problem.row_points)])
        end
        tmat = row_matrix(problem, trials, :t)
        nmat = row_matrix(problem, trials, :n)
        function spectrum(a)
            values = eigvals(Symmetric(Matrix(a * inverse_k * transpose(a))))
            largest = maximum(values)
            retained = filter(>(1e-10 * largest), values)
            return length(retained), largest / minimum(retained)
        end
        trank, tcond = spectrum(tmat)
        nrank, ncond = spectrum(nmat)
        totalrank, totalcond = spectrum([tmat; nmat])
        @printf("%s\t%d\t%d\t%.4g\t%d\t%.4g\t%d\t%.4g\t%d\n",
            string(method), length(trials), trank, tcond,
            nrank, ncond, totalrank, totalcond,
            length(problem.active_faces))
    end
end

function study(; resolutions = (16, 32, 64),
        densities = ((1.0, 1.0), (1000.0, 1.0), (1.0, 1000.0)),
        reconstruction_radius = 2.5, methods = nothing,
        soft_pressure_weight = 10.0, soft_cap = 1.0,
        soft_pressure_basis = :trigonometric)
    println("density\tn\tmethod\trows\tmean_support\tmax_amp\tself_comm_max\tcross_comm_max\tsmooth_comm\tfit_residual\tT_L2\tN_L2\tp_axis_L2\tp_diagonal_L2\tp_normal_L2\tT_Linf\tN_Linf\tp_axis_Linf\tp_diagonal_Linf\tp_normal_Linf")
    flush(stdout)
    results = NamedTuple[]
    for density in densities, n in resolutions
        options = isnothing(methods) ?
            (; reconstruction_radius, soft_pressure_weight, soft_cap,
                soft_pressure_basis) :
            (; reconstruction_radius, methods, soft_pressure_weight,
                soft_cap, soft_pressure_basis)
        for result in study_case(n, density; options...)
            push!(results, result)
            @printf("%s\t%d\t%s\t%d/%d\t%.2f\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n",
                string(density), n, string(result.method),
                result.rows, result.total_rows, result.support,
                result.max_amp, result.max_comm, result.cross_row_comm,
                result.smooth_comm,
                result.max_residual, result.l2..., result.linf...)
            flush(stdout)
        end
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    study()
end

end # module
