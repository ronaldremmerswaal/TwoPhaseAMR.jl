"""Focused old-state and perimeter checks for a copied rolling-flux module.

Include this in Main after `TwoPhaseAMR` and the adapted compact advection suite.
The compact suite supplies the independent `FrozenCellAdvectionWorkspace` and
`frozen_cell_advection_step!` reference definitions used below.
"""

using Test
using .TwoPhaseAMR

@testset "rolling regular-flux perimeter and promoted update" begin
    meshes = (
        # One leaf split into several column tiles at four/eight workers;
        # physical boundaries exercise every side of the perimeter union.
        BlockTreeMesh(base_mesh=(64, 32), block_size=(64, 32),
                      extent=(1.3, 0.9), periodic=(false, false)),
        BlockTreeMesh(base_mesh=(32, 64), block_size=(32, 64),
                      extent=(1.1, 0.7), periodic=(true, false)),
        # All-edge leaves test the narrow dimensions and mixed-level seams.
        BlockTreeMesh(base_mesh=(4, 32), block_size=(2, 16),
                      extent=(1.3, 0.9), periodic=(true, false), max_level=1,
                      refine=(key, origin, extent)->
                          key.level == 0 && key.i == 0),
        BlockTreeMesh(base_mesh=(32, 4), block_size=(16, 2),
                      extent=(1.1, 0.7), periodic=(false, true), max_level=1,
                      refine=(key, origin, extent)->
                          key.level == 0 && key.j == 0),
    )
    promotion_witnesses = 0
    for (case, mesh) in enumerate(meshes), T in (Float32, Float64)
        @test eltype(mesh.origin) === Float64
        if case <= 2 && Threads.nthreads(:default) > 1
            @test length(mesh.blocks) == 1
            @test mesh.block_size[2] > 1
            @test cld(mesh.block_size[2],
                      cld(mesh.block_size[2], 4Threads.nthreads(:default))) > 1
        elseif case > 2
            @test any(block -> block.key.level == 0, mesh.blocks)
            @test any(block -> block.key.level == 1, mesh.blocks)
        end
        scalar = CellField(mesh, T)
        initialize!(scalar, (x, y)->T(0.8 + 0.31sinpi(2x) -
                                      0.17cospi(2y) + 0.07sinpi(3x + y)))
        reference = copy(scalar)
        velocity = FaceVelocity(mesh, T; init=zero(T))
        observed_ws = AdvectionWorkspace(scalar)
        frozen_ws = FrozenCellAdvectionWorkspace(reference)
        boundary = Dirichlet((x, y, time)->T(0.4 + 0.13x - 0.27y + 0.21time))
        nx, ny = mesh.block_size
        h = mesh.halo
        for step in 1:4
            time = (step - 1) * 0.013
            set_velocities!(velocity, (x, y, t, axis)->
                T(axis == 1 ? 0.12 + 0.31sinpi(2x + t) - 0.19cospi(2y) :
                              -0.15 + 0.23cospi(2y - t) - 0.17sinpi(2x));
                time=time)
            # Direct face writes check that the copied path reads the live
            # owned values rather than a stale face-value cache.
            velocity.interior[firstindex(velocity.interior)] =
                T(-0.41 + 0.03step)
            velocity.interior[lastindex(velocity.interior)] =
                T(0.29 - 0.02step)
            if !isempty(velocity.boundary)
                velocity.boundary[firstindex(velocity.boundary)] =
                    T(-0.37 + 0.04step)
            end
            old = copy(reference)
            frozen_cell_advection_step!(reference, velocity, T(0.0037), frozen_ws;
                                        boundary, time)
            advect!(scalar, velocity, T(0.0037), observed_ws; boundary, time)
            for b in eachindex(mesh.blocks)
                @test isequal(interior(scalar, b), interior(reference, b))
                @test isequal(interior(observed_ws.increment, b),
                              interior(frozen_ws.increment, b))
                if T === Float32
                    volume = mesh.blocks[b].volume
                    offset = (b - 1) * nx * ny
                    for j in 1:ny, i in 1:nx
                        cell = offset + (j - 1) * nx + i
                        integrated = zero(T)
                        for pointer in frozen_ws.cell_offsets[cell]:(frozen_ws.cell_offsets[cell + 1] - 1)
                            face = frozen_ws.face_indices[pointer]
                            flux = frozen_ws.boundary_entries[pointer] ?
                                frozen_ws.boundary_flux[face] :
                                frozen_ws.interior_flux[face]
                            integrated += frozen_ws.face_signs[pointer] * flux
                        end
                        change = integrated / volume
                        stored = frozen_ws.increment.blocks[b][i + h, j + h]
                        @test isequal(stored, T(change))
                        original = old.blocks[b][i + h, j + h]
                        promotion_witnesses += !isequal(T(original + change),
                                                       T(original + stored))
                    end
                end
            end
        end
    end
    @test promotion_witnesses > 0
end
