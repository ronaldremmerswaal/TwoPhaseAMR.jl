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

@testset "Fixed-size PLIC centroid scratch" begin
    function reference_centroid(nx, ny, a)
        T = promote_type(typeof(nx), typeof(ny), typeof(a))
        half = T(0.5)
        tolerance = T(64) * eps(T)
        points = NTuple{2, T}[]
        for x in (-half, half)
            if abs(ny) > eps(T)
                y = (a - nx * x) / ny
                -half - tolerance <= y <= half + tolerance &&
                    push!(points, (x, clamp(y, -half, half)))
            end
        end
        for y in (-half, half)
            if abs(nx) > eps(T)
                x = (a - ny * y) / nx
                -half - tolerance <= x <= half + tolerance &&
                    push!(points, (clamp(x, -half, half), y))
            end
        end
        isempty(points) && return (zero(T), zero(T))
        unique!(points)
        return (
            sum(p[1] for p in points) / length(points),
            sum(p[2] for p in points) / length(points),
        )
    end
    for T in (Float32, Float64), nx in T.((-1, -0.5, -0.0, 0.0, 0.5, 1)),
            ny in T.((-1, -0.5, -0.0, 0.0, 0.5, 1)), a in T.((-1, -0.5, 0, 0.5, 1))
        @test isequal(
            TwoPhaseAMR._interface_centroid(nx, ny, a),
            reference_centroid(nx, ny, a)
        )
    end
end

@testset "Selectable PLIC reconstruction" begin
    mesh = BlockTreeMesh(block_size = (24, 24))
    levelset(x, y) = hypot(x - 0.5, y - 0.5) - 0.2
    lhf = InterfaceField(mesh, levelset; lipschitz = 1)
    lhf_ghf = InterfaceField(
        mesh, levelset; lipschitz = 1, reconstruction_method = :lhf_ghf,
    )
    wy = InterfaceField(
        mesh, levelset; lipschitz = 1, reconstruction_method = :wy,
    )
    @test lhf.reconstruction_method == :lhf
    @test lhf_ghf.reconstruction_method == :lhf_ghf
    @test wy.reconstruction_method == :wy
    @test copy(wy).reconstruction_method == :wy
    @test copy(lhf).reconstruction_method == :lhf
    @test copy(lhf_ghf).reconstruction_method == :lhf_ghf
    @test_throws ArgumentError InterfaceField(
        mesh; reconstruction_method = :unknown,
    )
    @test_throws ArgumentError InterfaceField(
        CellField(mesh); reconstruction_method = :unknown,
    )
    mixed = 0
    ghf_fallbacks = 0
    for ref in TwoPhaseAMR._cell_references(mesh)
        c = TwoPhaseAMR._get(wy.conditional_fraction, ref)
        0 < c < 1 || continue
        mixed += 1
        original_method = TwoPhaseAMR._get(lhf_ghf.reconstruction, ref)
        lhf_method = TwoPhaseAMR._get(lhf.reconstruction, ref)
        @test lhf_method in (1, 3, 5)
        @test isfinite(TwoPhaseAMR._get(lhf.curvature, ref))
        @test isfinite(TwoPhaseAMR._get(wy.curvature, ref))
        if original_method == 2
            ghf_fallbacks += 1
            @test lhf_method == 3
            @test TwoPhaseAMR._get(lhf.normal_x, ref) ==
                TwoPhaseAMR._get(wy.normal_x, ref)
            @test TwoPhaseAMR._get(lhf.normal_y, ref) ==
                TwoPhaseAMR._get(wy.normal_y, ref)
        elseif original_method in (1, 5)
            @test lhf_method == original_method
            @test TwoPhaseAMR._get(lhf.normal_x, ref) ==
                TwoPhaseAMR._get(lhf_ghf.normal_x, ref)
            @test TwoPhaseAMR._get(lhf.normal_y, ref) ==
                TwoPhaseAMR._get(lhf_ghf.normal_y, ref)
        end
        @test TwoPhaseAMR._get(wy.reconstruction, ref) == 3
        h = mesh.halo
        values = wy.conditional_fraction.blocks[ref.block]
        mx, my = TwoPhaseAMR._wy_normal(values, ref.i + h, ref.j + h)
        @test TwoPhaseAMR._get(wy.normal_x, ref) == mx
        @test TwoPhaseAMR._get(wy.normal_y, ref) == my
        @test TwoPhaseAMR._get(wy.intercept, ref) ≈
            TwoPhaseAMR._plane_intercept(c, mx, my)
    end
    @test mixed > 0
    @test ghf_fallbacks > 0
    reconstruct!(wy)
    @test all(
        TwoPhaseAMR._get(wy.reconstruction, ref) == 3
            for ref in TwoPhaseAMR._cell_references(mesh)
            if 0 < TwoPhaseAMR._get(wy.conditional_fraction, ref) < 1
    )
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
            reconstruction_method = :lhf_ghf,
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
        lipschitz = 1.0f0, reconstruction_method = :lhf_ghf,
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
        f = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2;
            lipschitz = 1, reconstruction_method = :lhf_ghf,
        )
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
        # for the symmetry case theta=pi/2). The touching cells carry the
        # contact-line curvature (cos θ_{3/2} - cos θ_w)/h of the first row
        # (techref eq:forces:contact-line), the mean curvature over the row.
        if angle_degrees == 160
            # NUMERICS CONCERN: at 160° the largest error of the contact-line
            # curvature, 0.265, 0.222, 0.190 at n = 32, 48, 64 (1/R = 3.33),
            # falls at the orders 0.44, 0.53; the contact-angle formulas gave
            # 0.435, 0.266, 0.203 (orders 1.21, 0.94), larger errors but
            # first order (cause not isolated; the line spreads over two to
            # four cells of the row).
            @test_broken all(rates .>= 0.7)
        else
            @test all(rates .>= 0.7)
        end
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

@testset "Contact-line curvature: energy consistency" begin
    # On the first row along a wall with a contact angle θ_w every cut cell
    # gets κ = (cos θ_{3/2} - cos θ_w)/h, θ_{3/2} the angle of the chord
    # between the wall-parallel heights of rows 1 and 2: the derivative of the
    # contact-line energy σ(|chord| - cos θ_w H₁) by the row's liquid volume
    # (techref eq:forces:contact-line). Along a closed loop of exact circular
    # caps (shift by h/2 at 90°, 90° → 60°, shift back, 60° → 90°; 200 steps,
    # h = 1/48) the trapezoidal work -Σ κ dV of the row of each contact line
    # vanishes: measured 2.7e-12 against a release of 0.0206 on the open path
    # 90° → 60°.
    TP = TwoPhaseAMR
    n, wall, steps = 48, pi / 3, 200
    h = 1 / n
    vertices(angle, x0; area = pi * 0.2^2 / 2) = (
        radius = sqrt(area / (angle - sin(angle) * cos(angle)));
        [(x0 + radius * sin(phi), radius * (cos(phi) - cos(angle))) for phi in range(angle, -angle; length = 801)]
    )
    cap(angle, x0) = PolygonIndicator(vertices(angle, x0))
    mesh = BlockTreeMesh(block_size = (n, n), periodic = (false, false))
    interface = InterfaceField(mesh, cap(pi / 2, 0.5); contact_angle = (nothing, nothing, wall, nothing))
    references = collect(TP._cell_references(mesh))
    row = [r for r in references if r.j == 1]
    sides = (r -> r.i <= n ÷ 2, r -> r.i > n ÷ 2)
    cut(field, r) = 0 < TP._conditional_fraction(field, r) < 1
    # Liquid volume and the curvatures of the cut cells of the row of each
    # contact line.
    function state(angle, x0)
        initialize!(interface, cap(angle, x0))
        reconstruct!(interface; force = true)
        return map(sides) do side
            cells = filter(side, row)
            volume = sum(TP._get(interface.fraction, r) for r in cells) * h^2
            volume, [TP._get(interface.curvature, r) for r in cells if cut(interface, r)]
        end
    end
    # Trapezoidal work of the rows and the largest relative spread of the
    # curvatures of a row (one curvature per row).
    function row_work(path)
        last = state(first(path)...)
        total = spread = 0.0
        for point in path[2:end]
            next = state(point...)
            for side in 1:2
                (before, kappa_before), (after, kappa_after) = last[side], next[side]
                spread = max(spread, (maximum(kappa_after) - minimum(kappa_after)) / maximum(abs, kappa_after))
                total -= (first(kappa_before) + first(kappa_after)) / 2 * (after - before)
            end
            last = next
        end
        return total, spread
    end
    angles = range(pi / 2, wall; length = steps + 1)
    shifts = range(0.5, 0.5 + h / 2; length = steps ÷ 4 + 1)
    release, release_spread = row_work([(angle, 0.5) for angle in angles])
    loop, loop_spread = row_work(
        vcat(
            [(pi / 2, x) for x in shifts], [(angle, last(shifts)) for angle in angles[2:end]],
            [(wall, x) for x in reverse(shifts)[2:end]], [(angle, 0.5) for angle in reverse(angles)[2:end]],
        )
    )
    @test max(release_spread, loop_spread) <= 1.0e-12
    @test 0.02 <= release <= 0.021
    @test abs(loop) <= 1.0e-10
    # At the 60° cap the cut cells of the row share one curvature per contact
    # line, close to the cap's 1/R = 3.13 (measured 3.17, the mean curvature over
    # the row).
    initialize!(interface, cap(wall, 0.5))
    reconstruct!(interface; force = true)
    cells = [r for r in row if cut(interface, r)]
    kappa = [TP._get(interface.curvature, r) for r in cells]
    for side in sides
        row_kappa = [k for (r, k) in zip(cells, kappa) if side(r)]
        @test length(row_kappa) == 2
        @test row_kappa[1] ≈ row_kappa[2] rtol = 1.0e-13
        @test row_kappa[1] ≈ sqrt(2 * pi / 3 - sqrt(3) / 2) / sqrt(pi * 0.2^2) rtol = 0.02
    end
    # Label symmetry: swapping the phases (C → 1 - C, θ_w → π - θ_w) flips the
    # sign of the row's curvature.
    function swapped_kappa(field, side, angle, cells)
        complement = copy(field.fraction)
        for b in eachindex(mesh.blocks)
            interior(complement, b) .= 1 .- interior(field.fraction, b)
        end
        swapped = InterfaceField(complement; contact_angle = ntuple(w -> w == side ? pi - angle : nothing, 4))
        return [TP._get(swapped.curvature, r) for r in cells]
    end
    @test swapped_kappa(interface, 3, wall, cells) ≈ -kappa rtol = 1.0e-10
    # Every wall: the cap mapped onto the left, right and top wall gives the
    # bottom row's curvatures in the first row of that wall (to the round-off
    # of the mapped vertices, measured 1e-12).
    function wall_row(points, side, angle)
        field = InterfaceField(mesh, PolygonIndicator(points); contact_angle = ntuple(w -> w == side ? angle : nothing, 4))
        in_row(r) = (r.i == 1, r.i == n, r.j == 1, r.j == n)[side]
        row_cells = [r for r in references if in_row(r) && cut(field, r)]
        return field, row_cells, [TP._get(field.curvature, r) for r in row_cells]
    end
    rotations = (1 => ((x, y),) -> (y, x), 2 => ((x, y),) -> (1 - y, x), 4 => ((x, y),) -> (x, 1 - y))
    for angle in (pi / 3, 2pi / 3)
        points = vertices(angle, 0.5 + 0.3h)
        bottom = sort(last(wall_row(points, 3, angle)))
        for (side, rotate) in rotations
            @test sort(last(wall_row(map(rotate, points), side, angle))) ≈ bottom rtol = 1.0e-10
        end
    end
    # At 35° on a smaller cap (area π0.1²/2) the line spreads over four to six
    # cells of the row and one or two of them have no valid columns (the row-2
    # transition leaves the window of four cells): they take the cosine of
    # the nearest valid cell of their run, so the row shares one curvature.
    small = 35pi / 180
    small_field, small_cells, small_kappa = wall_row(vertices(small, 0.5; area = pi * 0.1^2 / 2), 3, small)
    small_values = TP._fill_height_values!(small_field)
    @test count(small_cells) do r
        isnothing(TP._wall_row_cosine(small_values[r.block], r.i + 4, r.j + 4, 2, 1, 1.0))
    end > 0
    @test all(isfinite, small_kappa)
    @test maximum(small_kappa) - minimum(small_kappa) <= 1.0e-12 * maximum(abs, small_kappa)
    @test swapped_kappa(small_field, 3, small, small_cells) ≈ -small_kappa rtol = 1.0e-10
end

@testset "Contact-line curvature of a plane on rectangular cells" begin
    # A straight interface meeting a domain wall at the angle θ has the
    # contact-line curvature (cos θ - cos θ_w)/h exactly (h the cell size
    # normal to the wall): zero at Young's angle θ_w = θ. The chord slope
    # scales the tangential heights (in cells) by h_t/h; without the scale the
    # plane at Young's angle got an O(1/h) curvature on these 2:1 cells.
    TP = TwoPhaseAMR
    n = 32
    mesh = BlockTreeMesh(block_size = (n, n), extent = (1.0, 0.5), periodic = (false, false))
    hx, hy = mesh.blocks[1].spacing
    for theta in (pi / 3, pi / 2, 2pi / 3), offset in (0.0, 0.1)
        wall_angle = theta + offset
        # Bottom wall (normal y, h = hy), liquid left of the line through
        # (0.5 + 0.3hx, 0) that leaves the wall at θ in the liquid; the
        # transposed line on the left wall (normal x, h = hx).
        for (side, normal, intercept, h) in (
                (3, (sin(theta), cos(theta)), sin(theta) * (0.5 + 0.3hx), hy),
                (1, (cos(theta), sin(theta)), sin(theta) * (0.25 + 0.3hy), hx),
            )
            field = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1,
                contact_angle = ntuple(w -> w == side ? wall_angle : nothing, 4))
            initialize!(field, normal, intercept)
            reconstruct!(field; force = true)
            in_row(r) = side == 3 ? r.j == 1 : r.i == 1
            row = [r for r in TP._cell_references(mesh) if in_row(r) && 0 < TP._get(field.fraction, r) < 1]
            @test !isempty(row)
            for r in row
                @test TP._get(field.curvature, r) ≈ (cos(theta) - cos(wall_angle)) / h atol = 1.0e-8 / h
            end
        end
    end
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
        contact_angle = (nothing, nothing, theta, nothing),
        reconstruction_method = :lhf_ghf,
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
