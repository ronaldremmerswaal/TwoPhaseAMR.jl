@testset "phase-face scalar survives uniform and mixed AMR schedules" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
    qx = PhaseFaceScalar{1}(interface; liquid = 2.0, gas = 3.0)
    qy = PhaseFaceScalar{2}(interface; liquid = 5.0, gas = 7.0)
    amr = AMRWorkspace(mesh, interface, qx, qy)
    index = TwoPhaseAMR._find_field_buffer(amr, interface.fraction)
    strategy = amr.field_buffers[index].prolongation
    scalar_types = eltype(strategy.staggered)
    @test scalar_types !== Any
    @test all(isconcretetype, Base.uniontypes(scalar_types))
    @test fieldtype(typeof(qx), :interface) === typeof(interface)
    @test fieldtype(typeof(qy), :interface) === typeof(interface)
    @test qx isa scalar_types
    @test qy isa scalar_types
    register_adaptive_field!(amr, qx)
    @test length(strategy.staggered) == 2
    identities = (
        (qx, qx.liquid, qx.gas, qx.fraction,
            qx.workspace.tendency, qx.workspace.rates),
        (qy, qy.liquid, qy.gas, qy.fraction,
            qy.workspace.tendency, qy.workspace.rates),
    )
    phase_amounts() = (
        integrate(interface),
        integrate(qx, :liquid), integrate(qx, :gas),
        integrate(qy, :liquid), integrate(qy, :gas),
    )
    previous_amounts = Ref(phase_amounts())
    function check_amounts!()
        current = phase_amounts()
        # Use the integration vortex test's per-step roundoff allowance:
        # one conservative update or transfer contributes at most one
        # face-count-scaled amount of summation error.
        budget = 16 * length(mesh.faces) * eps(Float64)
        for (before, after) in zip(previous_amounts[], current)
            @test abs(after - before) <= budget * max(1.0, abs(before))
        end
        previous_amounts[] = current
    end
    function check_constants()
        for (q, axis, lvalue, gvalue) in ((qx, 1, 2.0, 3.0), (qy, 2, 5.0, 7.0))
            @test all(x -> isapprox(x, lvalue; atol = 2e-12), q.liquid.interior)
            @test all(x -> isapprox(x, gvalue; atol = 2e-12), q.gas.interior)
            unit = PhaseFaceScalar{axis}(interface)
            @test integrate(q, :liquid) ≈ lvalue * integrate(unit, :liquid) atol = 3e-12
            @test integrate(q, :gas) ≈ gvalue * integrate(unit, :gas) atol = 3e-12
        end
    end
    function check_state(expected_schedule)
        check_amounts!()
        for (q, saved) in ((qx, identities[1]), (qy, identities[2]))
            @test q === saved[1]
            @test q.interface === interface
            @test q.liquid === saved[2]
            @test q.gas === saved[3]
            @test q.fraction === saved[4]
            @test q.workspace.tendency === saved[5]
            @test q.workspace.rates === saved[6]
            @test q.workspace.schedule isa expected_schedule
            @test length(q.liquid.interior) == length(q.fraction.interior)
        end
        check_constants()
        velocity = FaceVelocity(mesh)
        work = WYAdvectionWorkspace(interface)
        entering_values = (
            copy(qx.liquid.interior), copy(qx.gas.interior),
            copy(qy.liquid.interior), copy(qy.gas.interior),
        )
        advect!(interface, velocity, 0.001, work; staggered_scalars = (qx, qy))
        @test qx.liquid.interior == entering_values[1]
        @test qx.gas.interior == entering_values[2]
        @test qy.liquid.interior == entering_values[3]
        @test qy.gas.interior == entering_values[4]
        check_amounts!()

        # Integrated streamfunction differences produce a nonzero periodic
        # face carrier with exactly cancelling fluxes at every cell, even
        # along a coarse/fine interface.
        psi(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
        for (i, face) in pairs(mesh.faces)
            x, y = face.center
            h = face.area / 2
            velocity.interior[i] = face.axis == 1 ?
                (psi(x, y + h) - psi(x, y - h)) / face.area :
                -(psi(x + h, y) - psi(x - h, y)) / face.area
        end
        @test maximum(abs, velocity.interior) > 0.01
        divergence = CellField(mesh)
        divergence!(divergence, velocity)
        @test maximum(maximum(abs, interior(divergence, b)) for b in 1:nblocks(mesh)) <= 2e-13
        advect!(interface, velocity, 0.001, work; staggered_scalars = (qx, qy))
        check_constants()
        check_amounts!()
    end
    U = TwoPhaseAMR.UniformDualAdvectionSchedule
    A = TwoPhaseAMR.AMRDualAdvectionSchedule
    @test nblocks(mesh) == 1
    check_state(U)
    @test adapt!(amr, [:refine]; step = 1, force = true).changed
    @test nblocks(mesh) == 4
    check_state(U)

    first_parent = first(mesh.blocks).key
    labels = fill(:keep, nblocks(mesh))
    labels[1] = :refine
    @test adapt!(amr, labels; step = 2, force = true).changed
    @test nblocks(mesh) == 7
    check_state(A)
    # Register both axes after the topology becomes mixed. Their constructor
    # schedule types differ from the earlier uniform scalars, but all four
    # remain in the same typed registry through later regrids.
    mixed_x = PhaseFaceScalar{1}(interface)
    mixed_y = PhaseFaceScalar{2}(interface)
    @test mixed_x.workspace.schedule isa A
    @test mixed_y.workspace.schedule isa A
    @test typeof(mixed_x) !== typeof(qx)
    @test typeof(mixed_y) !== typeof(qy)
    register_adaptive_field!(amr, mixed_x)
    register_adaptive_field!(amr, mixed_y)
    @test length(strategy.staggered) == 4
    @test mixed_x isa scalar_types
    @test mixed_y isa scalar_types
    first_keys = Set(block.key for block in mesh.blocks)
    first_schedule = qx.workspace.schedule

    # Coarsen one four-child family and refine a different level-one leaf in
    # one update: leaf count stays seven, but connectivity must be refreshed.
    labels = fill(:keep, nblocks(mesh))
    for (i, block) in pairs(mesh.blocks)
        if block.key.level == 2
            labels[i] = :coarsen
        elseif block.key.level == 1 && block.key != first_parent
            labels[i] = :refine
            break
        end
    end
    # Mark all four children, independent of mesh iteration order.
    for (i, block) in pairs(mesh.blocks)
        block.key.level == 2 && (labels[i] = :coarsen)
    end
    @test adapt!(amr, labels; step = 3, force = true).changed
    @test nblocks(mesh) == 7
    @test Set(block.key for block in mesh.blocks) != first_keys
    @test qx.workspace.schedule !== first_schedule
    check_state(A)

    labels = [block.key.level == 2 ? :coarsen : :keep for block in mesh.blocks]
    @test adapt!(amr, labels; step = 4, force = true).changed
    @test nblocks(mesh) == 4
    check_state(U)
    @test adapt!(amr, fill(:coarsen, nblocks(mesh)); step = 5, force = true).changed
    @test nblocks(mesh) == 1
    check_state(U)
end
