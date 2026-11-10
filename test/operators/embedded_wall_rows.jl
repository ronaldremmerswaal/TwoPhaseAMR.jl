using SparseArrays: spdiagm

@testset "grid-aligned embedded-wall Helmholtz convergence (#106)" begin
    L = 8.0
    exact(x, y) = sinpi(x / L) * sinpi((y - 1) / (L - 1))
    laplacian_factor = -((pi / L)^2 + (pi / (L - 1))^2)

    function wall_errors(n, refined)
        mesh = BlockTreeMesh(;
            base_mesh = (n, n), block_size = (4, 4), extent = (L, L),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) ->
            refined && key.level == 0 && key.i >= div(n, 8),
        )
        geometry = GeometryField(mesh, (x, y) -> y - 1; lipschitz = 1)
        operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
        wall_faces = [
            i for (i, face) in pairs(mesh.faces)
                if face.axis == 2 && face.center[2] == 1
        ]
        @test !isempty(wall_faces)
        @test isempty(intersect(wall_faces, operator.y.local_to_global))

        errors = NTuple{3, Float64}[]
        for (axis, component) in enumerate((operator.x, operator.y))
            @test component.stiffness ≈ transpose(component.stiffness) atol = 1.0e-12
            points = [mesh.faces[g].center for g in component.local_to_global]
            open = [geometry.face_dual_volumes[g] > 0 for g in component.local_to_global]
            if axis == 1
                h = L / n
                near = [
                    i for i in eachindex(points)
                        if open[i] && 1 < points[i][2] < 1 + h &&
                        2h < points[i][1] < L - 2h &&
                        (!refined || abs(points[i][1] - L / 2) > 2h)
                ]
                @test !isempty(near)
                linear = [open[i] ? points[i][2] - 1 : 0.0 for i in eachindex(points)]
                @test maximum(abs, (component.stiffness * linear)[near]) <= 1.0e-12
            end
            exact_values = [open[i] ? exact(points[i]...) : 0.0 for i in eachindex(points)]
            rhs = (1 - laplacian_factor) .* exact_values
            matrix = spdiagm(0 => ones(length(rhs))) - component.laplacian
            numerical = matrix \ rhs
            error = (numerical - exact_values)[open]
            volume = geometry.face_dual_volumes[component.local_to_global][open]
            push!(
                errors,
                (
                    maximum(abs, error),
                    sqrt(sum(volume .* error .^ 2) / sum(volume)),
                    sum(volume .* abs.(error)) / sum(volume),
                ),
            )
        end
        return errors
    end

    for refined in (false, true)
        errors = [wall_errors(n, refined) for n in (16, 32, 64)]
        for axis in 1:2, norm in 1:3, level in 1:2
            order = log2(errors[level][axis][norm] / errors[level + 1][axis][norm])
            @test order >= (norm == 1 ? 1.5 : 1.8)
        end
    end
end

@testset "aligned normal faces are prescribed for both slip conditions (#106)" begin
    for normal in 1:2, side in (-1, 1), refined in (false, true)
        mesh = BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) ->
            refined && key.level == 0 && key.i >= 1,
        )
        geometry = GeometryField(
            mesh, (x, y) -> side * ((normal == 1 ? x : y) - 4); lipschitz = 1,
        )
        wall_faces = [
            i for (i, face) in pairs(mesh.faces)
                if face.axis == normal && face.center[normal] == 4
        ]
        @test !isempty(wall_faces)
        for boundary in (NoSlip(), FreeSlip())
            operator = StaggeredDiffusionOperator(geometry; boundary)
            component = normal == 1 ? operator.x : operator.y
            @test isempty(intersect(wall_faces, component.local_to_global))
            @test component.stiffness ≈ transpose(component.stiffness) atol = 1.0e-12
            if !refined
                tangent = 3 - normal
                tangential_component = tangent == 1 ? operator.x : operator.y
                face_index = findfirst(
                    face -> face.axis == tangent && face.center[tangent] == 2 &&
                        face.center[normal] == 4 + side / 2,
                    mesh.faces,
                )
                @test !isnothing(face_index)
                local_index = findfirst(==(face_index), tangential_component.local_to_global)
                @test !isnothing(local_index)
                conductance = sum(
                    wall.conductance for wall in tangential_component.wall_connections
                        if iszero(wall.boundary_axis) && wall.tangential &&
                        wall.degree_of_freedom == local_index;
                    init = 0.0,
                )
                # Unit wall length at distance h/2 = 1/2.
                @test conductance ≈ 2.0 atol = 1.0e-12
            end

            velocity = FaceVelocity(mesh, (x, y, time, axis) -> 1.0)
            clean = copy(velocity)
            velocity.interior[wall_faces] .= 13.0
            clean.interior[wall_faces] .= 0.0
            tendency = FaceVelocity(mesh, (x, y, time, axis) -> 17.0)
            reference = FaceVelocity(mesh)
            diffusion!(tendency, velocity, operator)
            diffusion!(reference, clean, operator)
            @test all(iszero, tendency.interior[wall_faces])
            @test tendency.interior[component.local_to_global] ≈
                reference.interior[component.local_to_global] atol = 1.0e-12

            solver = StaggeredEllipticSolver(
                StaggeredEllipticOperator(operator); preconditioner = nothing,
            )
            solution = copy(velocity)
            rhs = copy(clean)
            rhs.interior[wall_faces] .= 19.0
            stats = solve!(solution, solver, rhs; tolerance = 1.0e-10, max_cycles = 1000)
            @test stats.converged
            @test all(iszero, solution.interior[wall_faces])
            if normal == 2 && side == 1 && refined && boundary isa NoSlip
                multigrid = StaggeredEllipticSolver(StaggeredEllipticOperator(operator))
                multigrid_solution = copy(velocity)
                multigrid_stats = solve!(
                    multigrid_solution, multigrid, rhs;
                    tolerance = 1.0e-10, max_cycles = 1000,
                )
                @test multigrid_stats.converged
                @test all(iszero, multigrid_solution.interior[wall_faces])
                @test multigrid_solution.interior ≈ solution.interior atol = 1.0e-8
            end
        end
    end
end

@testset "Cartesian wall-normal distance on anisotropic grids (#106)" begin
    for T in (Float32, Float64), normal in 1:2, side in (-1, 1)
        extent = (T(8), T(16))
        spacing = (T(1), T(2))
        wall = extent[normal] / 2
        tangent = 3 - normal
        mesh = BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (4, 4),
            origin = (zero(T), zero(T)), extent, periodic = (false, false),
        )
        geometry = GeometryField(
            mesh, (x, y) -> T(side) * ((normal == 1 ? x : y) - wall);
            lipschitz = one(T),
        )
        operator = StaggeredDiffusionOperator(geometry; viscosity = one(T))
        component = tangent == 1 ? operator.x : operator.y
        face_index = findfirst(
            face -> face.axis == tangent &&
                face.center[tangent] == 2spacing[tangent] &&
                face.center[normal] == wall + side * spacing[normal] / 2,
            mesh.faces,
        )
        @test !isnothing(face_index)
        local_index = findfirst(==(face_index), component.local_to_global)
        @test !isnothing(local_index)
        conductance = sum(
            connection.conductance for connection in component.wall_connections
                if iszero(connection.boundary_axis) && connection.tangential &&
                connection.degree_of_freedom == local_index;
            init = zero(T),
        )
        @test conductance ≈ 2spacing[tangent] / spacing[normal] atol = 64eps(T)
        @test eltype(component.stiffness) === T
        @test component.stiffness == transpose(component.stiffness)
    end
end

@testset "inclined embedded-wall Helmholtz comparison (#106)" begin
    # The pre-existing general PLIC wall closure is not second order in this
    # problem. Record its observed convergence and prevent its error growing
    # while the Cartesian-wall special case is corrected.
    L = 8.0
    slope = 0.1
    function exact_and_laplacian(x, y)
        wall = 1 + slope * x
        a = pi * x / L
        b = pi * (y - wall) / (L - wall)
        ax = pi / L
        bx = pi * slope * (y - L) / (L - wall)^2
        by = pi / (L - wall)
        bxx = 2pi * slope^2 * (y - L) / (L - wall)^3
        value = sin(a) * sin(b)
        laplacian = -(ax^2 + bx^2 + by^2) * value +
            2ax * cos(a) * cos(b) * bx + sin(a) * cos(b) * bxx
        return value, laplacian
    end

    function inclined_errors(n, refined)
        mesh = BlockTreeMesh(;
            base_mesh = (n, n), block_size = (4, 4), extent = (L, L),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) ->
            refined && key.level == 0 && key.i >= div(n, 8),
        )
        geometry = GeometryField(
            mesh, (x, y) -> y - 1 - slope * x; lipschitz = hypot(1, slope),
        )
        operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
        errors = NTuple{3, Float64}[]
        for component in (operator.x, operator.y)
            points = [mesh.faces[g].center for g in component.local_to_global]
            open = [geometry.face_dual_volumes[g] > 0 for g in component.local_to_global]
            values = [exact_and_laplacian(point...) for point in points]
            exact = [open[i] ? values[i][1] : 0.0 for i in eachindex(points)]
            rhs = [open[i] ? values[i][1] - values[i][2] : 0.0 for i in eachindex(points)]
            numerical = (spdiagm(0 => ones(length(rhs))) - component.laplacian) \ rhs
            error = (numerical - exact)[open]
            volume = geometry.face_dual_volumes[component.local_to_global][open]
            push!(
                errors,
                (
                    maximum(abs, error),
                    sqrt(sum(volume .* error .^ 2) / sum(volume)),
                    sum(volume .* abs.(error)) / sum(volume),
                ),
            )
        end
        return errors
    end

    # Finest-level errors on the unmodified PLIC operator; a 2% allowance
    # covers numerical variation without hiding a degradation of the closure.
    baseline = (
        ((0.05179795, 0.00562358, 0.00270754), (0.06648078, 0.00300688, 0.00129481)),
        ((0.05144898, 0.00437488, 0.00202357), (0.06636646, 0.00226386, 0.00094285)),
    )
    for (mesh_index, refined) in enumerate((false, true))
        errors = [inclined_errors(n, refined) for n in (16, 32, 64)]
        for axis in 1:2, norm in 1:3
            @test errors[end][axis][norm] <= 1.02 * baseline[mesh_index][axis][norm]
            for level in 1:2
                order = log2(errors[level][axis][norm] / errors[level + 1][axis][norm])
                @test order >= (norm == 1 ? 0.6 : 0.8)
            end
        end
    end
end
