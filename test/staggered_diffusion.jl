using LinearAlgebra: Symmetric, eigvals

function _face_component_energy(velocity, tendency, component)
    result = 0.0
    for local_index in eachindex(component.local_to_global)
        global_index = component.local_to_global[local_index]
        result += component.volumes[local_index] *
                  velocity.interior[global_index] *
                  tendency.interior[global_index]
    end
    return result
end

@testset "staggered diffusion is dissipative on uniform and AMR meshes" begin
    meshes = (
        BlockTreeMesh(block_size=(8, 8)),
        BlockTreeMesh(block_size=(8, 8), periodic=(false, false)),
        BlockTreeMesh(
            block_size=(4, 4), periodic=(false, false), max_level=2,
            refine=(key, origin, extent) ->
                key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes, boundary in (NoSlip(), FreeSlip())
        operator = StaggeredDiffusionOperator(mesh; viscosity=0.2, boundary)
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) ->
                axis == 1 ? sinpi(2x) + 0.3cospi(2y) :
                            cospi(2x) - 0.2sinpi(4y))
        tendency = FaceVelocity(
            mesh, (x, y, time, axis) -> 0.0)
        diffusion!(tendency, velocity, operator)
        energy_rate = _face_component_energy(velocity, tendency, operator.x) +
                      _face_component_energy(velocity, tendency, operator.y)
        @test energy_rate <= 256eps() * length(velocity.interior)
        @test operator.x.stiffness ≈ transpose(operator.x.stiffness)
        @test operator.y.stiffness ≈ transpose(operator.y.stiffness)
        @test minimum(eigvals(Symmetric(Matrix(operator.x.stiffness)))) >= -2e-12
        @test minimum(eigvals(Symmetric(Matrix(operator.y.stiffness)))) >= -2e-12
        @test stable_timestep(operator) > 0
    end
end

@testset "wall slip and no-slip closures" begin
    mesh = BlockTreeMesh(
        block_size=(12, 12), periodic=(true, false), extent=(1.0, 1.0))
    couette = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? y : 0.0)
    tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    moving_lid = NoSlip((x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0)
    no_slip = StaggeredDiffusionOperator(mesh; boundary=moving_lid)
    diffusion!(tendency, couette, no_slip)
    @test maximum(abs, tendency.interior) <= 2e-12

    constant_tangent = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 2.0 : 0.0)
    free_slip = StaggeredDiffusionOperator(mesh; boundary=FreeSlip())
    diffusion!(tendency, constant_tangent, free_slip)
    @test maximum(abs, tendency.interior) <= 2e-12

    stationary = StaggeredDiffusionOperator(mesh)
    diffusion!(tendency, constant_tangent, stationary)
    @test minimum(tendency.interior) < 0
end

@testset "uniform periodic staggered Laplacian convergence" begin
    function error(nx)
        mesh = BlockTreeMesh(block_size=(nx, nx))
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
        tendency = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        operator = StaggeredDiffusionOperator(mesh; boundary=FreeSlip())
        diffusion!(tendency, velocity, operator)
        squared = 0.0
        volume = 0.0
        for component in (operator.x, operator.y)
            for local_index in eachindex(component.local_to_global)
                global_index = component.local_to_global[local_index]
                face = mesh.faces[global_index]
                x, y = face.center
                exact = -8pi^2 * (face.axis == 1 ?
                    sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y))
                squared += component.volumes[local_index] *
                           (tendency.interior[global_index] - exact)^2
                volume += component.volumes[local_index]
            end
        end
        return sqrt(squared / volume)
    end
    errors = error.((8, 16, 32))
    @test log2(errors[1] / errors[2]) > 1.9
    @test log2(errors[2] / errors[3]) > 1.9
end
