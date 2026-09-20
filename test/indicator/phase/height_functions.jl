using Test, TwoPhaseAMR
function circle_fraction(xlo, xhi, ylo, yhi, r)
    max(abs(xlo), abs(xhi))^2 + max(abs(ylo), abs(yhi))^2 <= r * r && return 1.0
    minx = xlo > 0 ? xlo : xhi < 0 ? -xhi : 0.0
    miny = ylo > 0 ? ylo : yhi < 0 ? -yhi : 0.0
    minx^2 + miny^2 >= r * r && return 0.0
    lo, hi = max(xlo, -r), min(xhi, r)
    lo >= hi && return 0.0
    cuts = [lo, hi]
    for y in (ylo, yhi)
        if abs(y) < r
            root = sqrt(r * r - y * y)
            for x in (-root, root)
                lo < x < hi && push!(cuts, x)
            end
        end
    end
    sort!(cuts)
    primitive(x) = (x * sqrt(max(0.0, r * r - x * x)) + r * r * asin(clamp(x / r, -1, 1))) / 2
    area = 0.0
    for (a, b) in zip(cuts[1:(end - 1)], cuts[2:end])
        mid = (a + b) / 2; h = sqrt(max(0.0, r * r - mid * mid))
        min(yhi, h) > max(ylo, -h) || continue
        upper = yhi < h ? yhi * (b - a) : primitive(b) - primitive(a)
        lower = ylo > -h ? ylo * (b - a) : -primitive(b) + primitive(a)
        area += upper - lower
    end
    return clamp(area / ((xhi - xlo) * (yhi - ylo)), 0.0, 1.0)
end

@testset "Wall PLIC normals use a contact-angle-free two-point stencil" begin
    mesh = BlockTreeMesh(
        block_size = (64, 64), periodic = (false, false),
    )
    for angle in (0.1, 0.4, 0.7, 1.2, 2.0, 3.0, 4.0, 5.0)
        normal = (cos(angle), sin(angle))
        interface = InterfaceField(
            mesh,
            (x, y) -> normal[1] * (x - 0.5) +
                normal[2] * (y - 0.5);
            lipschitz = 1,
        )
        wall_cells = CellRef[]
        for ref in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(interface.fraction, ref)
            1.0e-10 < c < 1 - 1.0e-10 || continue
            x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
            h = mesh.blocks[ref.block].spacing
            min(x, 1 - x) <= h[1] / 2 + 64eps() ||
                min(y, 1 - y) <= h[2] / 2 + 64eps() || continue
            push!(wall_cells, ref)
        end
        @test !isempty(wall_cells)
        normals_match = true
        reconstructed = true
        for ref in wall_cells
            hx, hy = mesh.blocks[ref.block].spacing
            nx = TwoPhaseAMR._get(interface.normal_x, ref) / hx
            ny = TwoPhaseAMR._get(interface.normal_y, ref) / hy
            scale = hypot(nx, ny)
            normals_match &= isapprox(
                [nx / scale, ny / scale], collect(normal); atol = 2.0e-12,
            )
            reconstructed &=
                TwoPhaseAMR._get(interface.reconstruction, ref) in (2, 5)
        end
        @test normals_match
        @test reconstructed
    end
end
@testset "WY fallback exactly reconstructs linear interfaces" begin
    all_exact = true
    for angle in range(0.02, 2pi - 0.02; length = 60),c in (0.03, 0.4, 0.6, 0.97)
        n = (cos(angle), sin(angle));a = TwoPhaseAMR._plane_intercept(c, n...)
        v = [TwoPhaseAMR._plane_fraction(n..., a, i - 0.5, i + 0.5, j - 0.5, j + 0.5) for i in -1:1,j in -1:1]
        mx, my = TwoPhaseAMR._wy_normal(v, 2, 2)
        norm = hypot(mx, my)
        all_exact &= isapprox(
            [mx / norm, my / norm], collect(n); atol = 2.0e-13,
        )
    end
    @test all_exact
end
@testset "LHF planar normals and curvature" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (16, 16), max_level = 2, periodic = (false, false),
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        if kind == 3
            seed = CellField(mesh)
            amr = AMRWorkspace(mesh, seed)
            adapt!(amr, [:refine]; step = 1, force = true)
        end
        for angle in (0.1, 0.4, 0.7, 1.2, 2.0, 3.0, 4.0, 5.0)
            n = (cos(angle), sin(angle))
            intercept = sum(n) * 0.5
            f = InterfaceField(
                mesh,
                (x, y) -> n[1] * x + n[2] * y - intercept;
                lipschitz = 1
            )
            normals_match = true
            reconstructions_match = true
            curvature_is_flat = true
            for r in TwoPhaseAMR._cell_references(mesh)
                c = TwoPhaseAMR._get(f.fraction, r)
                1.0e-10 < c < 1 - 1.0e-10 || continue
                x, y = cellcenter(mesh, r.block, r.i, r.j)
                0.2 < x < 0.8 && 0.2 < y < 0.8 || continue
                hx, hy = mesh.blocks[r.block].spacing
                mx = TwoPhaseAMR._get(f.normal_x, r) / hx;my = TwoPhaseAMR._get(f.normal_y, r) / hy
                norm = hypot(mx, my)
                normals_match &= isapprox(
                    [mx / norm, my / norm], collect(n); atol = 1.0e-10,
                )
                reconstructions_match &=
                    TwoPhaseAMR._get(f.reconstruction, r) == 1
                curvature_is_flat &=
                    abs(TwoPhaseAMR._get(f.curvature, r)) < 1.0e-8
            end
            @test normals_match
            @test reconstructions_match
            @test curvature_is_flat
        end
    end
end

@testset "Physical interface height" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (8, 8),
        origin = (-1.0, -2.0), extent = (2.0, 4.0),
        periodic = (true, false), max_level = 1,
        refine = (key, origin, extent) ->
        key.level == 0 && key.i == 1 && key.j == 0,
    )

    liquid_below = InterfaceField(mesh)
    initialize!(liquid_below, (-0.2, 1.0), 0.25)
    @test interface_height(liquid_below) ≈ 0.25 atol = 2.0e-12
    @test interface_height(liquid_below; position = 0.4) ≈ 0.33 atol = 2.0e-12
    @test interface_height(liquid_below; position = 2.4) ≈ 0.33 atol = 2.0e-12

    liquid_above = InterfaceField(mesh)
    initialize!(liquid_above, (0.2, -1.0), -0.25)
    @test interface_height(liquid_above) ≈ 0.25 atol = 2.0e-12
    @test interface_height(liquid_above; position = -0.6) ≈ 0.13 atol = 2.0e-12

    liquid_left = InterfaceField(mesh)
    initialize!(liquid_left, (1.0, 0.1), 0.4)
    @test interface_height(
        liquid_left; direction = :x, position = -0.5,
    ) ≈ 0.45 atol = 2.0e-12

    @test_throws ArgumentError interface_height(
        liquid_below; direction = :diagonal,
    )
    @test_throws ArgumentError interface_height(
        liquid_below; position = Inf,
    )
end

@testset "Interface geometry preserves Float32 arithmetic" begin
    mesh = BlockTreeMesh(
        block_size = (32, 32), origin = (0.0f0, 0.0f0),
        extent = (1.0f0, 1.0f0), periodic = (false, false)
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.5f0, y - 0.5f0) - 0.2f0;
        lipschitz = 1.0f0
    )
    @test eltype(interface) === Float32
    mixed = [
        ref for ref in TwoPhaseAMR._cell_references(mesh)
            if 0 < TwoPhaseAMR._get(interface.fraction, ref) < 1
    ]
    @test !isempty(mixed)
    @test all(
        TwoPhaseAMR._get(interface.reconstruction, ref) in (1, 2) &&
            isfinite(TwoPhaseAMR._get(interface.curvature, ref))
            for ref in mixed
    )
    @test TwoPhaseAMR._interface_centroid(0.3f0, -0.7f0, 0.1f0) isa
        NTuple{2, Float32}
end
@testset "Circular LHF curvature converges" begin
    errors = Float64[]
    ghf_counts = Int[]
    resolutions = (32, 48, 64)
    for n in resolutions
        mesh = BlockTreeMesh(block_size = (n, n), periodic = (false, false))
        f = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2; lipschitz = 1)
        e = Float64[]
        ghf = 0
        valid_reconstructions = true
        finite_curvature = true
        for r in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(f.fraction, r)
            1.0e-8 < c < 1 - 1.0e-8 || continue
            method = TwoPhaseAMR._get(f.reconstruction, r)
            valid_reconstructions &= method in (1, 2)
            ghf += method == 2
            k = TwoPhaseAMR._get(f.curvature, r)
            finite_curvature &= isfinite(k)
            method == 1 && push!(e, abs(k - 5.0))
        end
        @test valid_reconstructions
        @test finite_curvature
        push!(errors, maximum(e))
        push!(ghf_counts, ghf)
    end
    rates = [
        log(errors[index] / errors[index + 1]) /
            log(resolutions[index + 1] / resolutions[index])
            for index in 1:(length(resolutions) - 1)
    ]
    @test all(rates .>= 2 - 0.3)
    @test any(>(0), ghf_counts)
end

@testset "Curvature is bounded by the local mesh resolution" begin
    n = 32
    mesh = BlockTreeMesh(
        block_size = (n, n), periodic = (false, false),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.04;
        lipschitz = 1,
    )
    finite_curvatures = [
        TwoPhaseAMR._get(interface.curvature, ref)
            for ref in TwoPhaseAMR._cell_references(mesh)
            if isfinite(TwoPhaseAMR._get(interface.curvature, ref))
    ]
    @test !isempty(finite_curvatures)
    h = maximum(mesh.blocks[1].spacing)
    @test maximum(abs, finite_curvatures) <= inv(2h)
    @test TwoPhaseAMR._clip_curvature_to_mesh(100.0, (h, h)) == inv(2h)
    @test TwoPhaseAMR._clip_curvature_to_mesh(-100.0, (h, h)) == -inv(2h)
end

@testset "Curvature AMR detects invalid heights and filaments" begin
    plane = zeros(11, 11)
    plane[1:5, :] .= 1
    plane[6, :] .= 0.5
    valid, filament = TwoPhaseAMR._lhf_refinement_status(plane, 6, 6)
    @test valid
    @test !filament

    thin = zeros(11, 11)
    thin[5:7, :] .= 1
    thin[5, :] .= 0.25
    thin[7, :] .= 0.25
    valid, filament = TwoPhaseAMR._lhf_refinement_status(thin, 6, 6)
    @test !valid
    @test filament

    unresolved = fill(0.5, 11, 11)
    valid, filament = TwoPhaseAMR._lhf_refinement_status(
        unresolved, 6, 6
    )
    @test !valid
    @test !filament
end

@testset "Contact-angle curvature converges at triple points" begin
    for angle_degrees in (20, 55, 90, 125, 160)
        errors = Float64[]
        resolutions = (32, 48, 64)
        for n in resolutions
            theta = deg2rad(angle_degrees)
            radius = 0.3
            center_x = 0.027
            center_y = -radius * cos(theta)
            mesh = BlockTreeMesh(
                block_size = (n, n), origin = (-0.5, 0.0),
                extent = (1.0, 1.0), periodic = (false, false)
            )
            interface = InterfaceField(
                mesh,
                (x, y) -> hypot(x - center_x, y - center_y) - radius;
                lipschitz = 1,
                contact_angle = (nothing, nothing, theta, nothing)
            )
            local_errors = Float64[]
            for ref in TwoPhaseAMR._cell_references(mesh)
                _, y = cellcenter(mesh, ref.block, ref.i, ref.j)
                y <= 0.5 / n + 64eps() || continue
                TwoPhaseAMR._plic_touches_wall(interface, ref, 3) || continue
                curvature = TwoPhaseAMR._get(interface.curvature, ref)
                isfinite(curvature) &&
                    push!(local_errors, abs(curvature - inv(radius)))
            end
            @test !isempty(local_errors)
            push!(errors, maximum(local_errors))
        end
        rates = [
            log(errors[index] / errors[index + 1]) /
                log(resolutions[index + 1] / resolutions[index])
                for index in 1:(length(resolutions) - 1)
        ]
        # The thesis predicts first order at the triple point (second order
        # for the symmetry case theta=pi/2).
        @test all(rates .>= 0.7)
        if angle_degrees == 90
            @test all(rates .>= 1.8)
        end
    end
end

@testset "Contact angle changes curvature but not the PLIC interface" begin
    theta = deg2rad(55)
    radius = 0.3
    mesh = BlockTreeMesh(
        block_size = (64, 64), origin = (-0.5, 0.0),
        extent = (1.0, 1.0), periodic = (false, false),
    )
    fraction = InterfaceField(
        mesh,
        (x, y) -> hypot(x, y + radius * cos(theta)) - radius;
        lipschitz = 1, contact_angle = nothing,
    ).fraction
    first = InterfaceField(
        copy(fraction);
        contact_angle = (nothing, nothing, theta, nothing),
    )
    second = InterfaceField(
        copy(fraction);
        contact_angle = (nothing, nothing, pi - theta, nothing),
    )
    curvature_changed = false
    same_interface = true
    for ref in TwoPhaseAMR._cell_references(mesh)
        c = TwoPhaseAMR._get(first.fraction, ref)
        0 < c < 1 || continue
        same_interface &= isapprox(
            TwoPhaseAMR._get(first.normal_x, ref),
            TwoPhaseAMR._get(second.normal_x, ref); atol = 32eps(),
        )
        same_interface &= isapprox(
            TwoPhaseAMR._get(first.normal_y, ref),
            TwoPhaseAMR._get(second.normal_y, ref); atol = 32eps(),
        )
        same_interface &= isapprox(
            TwoPhaseAMR._get(first.intercept, ref),
            TwoPhaseAMR._get(second.intercept, ref); atol = 32eps(),
        )
        _, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        y <= mesh.blocks[ref.block].spacing[2] / 2 + 64eps() || continue
        k1 = TwoPhaseAMR._get(first.curvature, ref)
        k2 = TwoPhaseAMR._get(second.curvature, ref)
        isfinite(k1) && isfinite(k2) &&
            (curvature_changed |= k1 != k2)
    end
    @test same_interface
    @test curvature_changed
end

@testset "Expanded GHF resolves a shallow wall arc" begin
    theta = deg2rad(20)
    radius = 0.2
    center_y = -radius * cos(theta)
    mesh = BlockTreeMesh(
        block_size = (48, 48), periodic = (false, false)
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.5, y - center_y) - radius;
        lipschitz = 1,
        contact_angle = (nothing, nothing, theta, nothing)
    )
    mixed = [
        ref for ref in TwoPhaseAMR._cell_references(mesh)
            if 0 < TwoPhaseAMR._get(interface.fraction, ref) < 1
    ]
    @test !isempty(mixed)
    @test any(
        TwoPhaseAMR._get(interface.reconstruction, ref) == 2
            for ref in mixed
    )
    @test all(
        isfinite(TwoPhaseAMR._get(interface.curvature, ref))
            for ref in mixed
    )
end
