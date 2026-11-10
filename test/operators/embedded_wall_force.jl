module EmbeddedWallForceFixture
    include(joinpath(@__DIR__, "..", "integration", "cylinder_reference_problem.jl"))
end

@testset "aligned embedded-wall force balances viscous momentum loss (#106)" begin
    # A compactly supported piecewise-linear profile has unit wall-normal
    # derivative. Its interior kinks produce no net integrated diffusion;
    # the opposite domain wall has zero velocity and zero adjacent values.
    for T in (Float32, Float64), normal in 1:2, side in (-1, 1), refined in (false, true)
        L = T(8)
        wall = L / 2
        tangent = 3 - normal
        mesh = BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (4, 4),
            origin = (zero(T), zero(T)), extent = (L, L),
            periodic = normal == 1 ? (false, true) : (true, false), max_level = 1,
            refine = (key, origin, extent) ->
            refined && key.level == 0 && origin[tangent] >= wall,
        )
        geometry = GeometryField(
            mesh, (x, y) -> T(side) * ((normal == 1 ? x : y) - wall);
            lipschitz = one(T),
        )
        pressure = CellField(mesh)
        viscosity = T(0.3)
        side_slip = VelocityBoundaryConditions(;
            left = FreeSlip(), right = FreeSlip(), bottom = FreeSlip(), top = FreeSlip(),
        )
        for boundary in (NoSlip(), FreeSlip(), side_slip)
            operator = StaggeredDiffusionOperator(geometry; viscosity, boundary)
            for axis in 1:2
                profile(x, y) = begin
                    d = T(side) * ((normal == 1 ? x : y) - wall)
                    max(zero(T), min(d, T(2) - d))
                end
                velocity = FaceVelocity(
                    mesh, (x, y, time, component) ->
                    component == axis ? profile(x, y) : zero(T),
                )
                rate = FaceVelocity(mesh)
                diffusion!(rate, velocity, operator)
                component = axis == 1 ? operator.x : operator.y
                momentum_loss = -sum(
                    component.volumes[i] * rate.interior[component.local_to_global[i]]
                        for i in eachindex(component.local_to_global);
                    init = zero(T),
                )
                coefficients = EmbeddedWallForceFixture.embedded_force_coefficients(
                    pressure, velocity, geometry, operator,
                )
                force = (axis == 1 ? coefficients.viscous_drag : coefficients.viscous_lift) / 2
                expected = boundary isa FreeSlip && axis == tangent ? zero(T) : viscosity * L
                tolerance = T(512) * eps(T) * L
                @test isapprox(momentum_loss, expected; atol = tolerance, rtol = tolerance)
                @test isapprox(force, momentum_loss; atol = tolerance, rtol = tolerance)
            end
        end
    end
end
