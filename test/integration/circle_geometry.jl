# Independent analytic reference: area inside both a disk and a half-plane,
# clipped to a cell rectangle. Breakpoints include every change of active bound.
function circle_plane_overlap(xlo, xhi, ylo, yhi, nx, ny, a; radius = 0.15)
    r = radius
    lo, hi = max(xlo, -r), min(xhi, r)
    lo >= hi && return 0.0
    if abs(ny) < 1.0e-14hypot(nx, ny)
        if nx > 0
            hi = min(hi, a / nx)
        elseif nx < 0
            lo = max(lo, a / nx)
        elseif a < 0
            return 0.0
        end
        lo >= hi && return 0.0
    end
    cuts = [lo, hi]
    for y in (ylo, yhi)
        if abs(y) < r
            root = sqrt(r * r - y * y)
            append!(cuts, (-root, root))
        end
        nx != 0 && push!(cuts, (a - ny * y) / nx)
    end
    norm2 = nx * nx + ny * ny
    if norm2 > 0 && a * a < r * r * norm2
        root = sqrt(r * r * norm2 - a * a)
        append!(cuts, ((a * nx - ny * root) / norm2, (a * nx + ny * root) / norm2))
    end
    filter!(x -> lo <= x <= hi, cuts);sort!(cuts);unique!(cuts)
    primitive(x) = (x * sqrt(max(0.0, r * r - x * x)) + r * r * asin(clamp(x / r, -1, 1))) / 2
    area = 0.0
    for (left, right) in zip(cuts[1:(end - 1)], cuts[2:end])
        mid = (left + right) / 2;root = sqrt(max(0.0, r * r - mid * mid))
        width = right - left;circle = primitive(right) - primitive(left)
        upper, ui = yhi, yhi * width
        lower, li = ylo, ylo * width
        root < upper && ((upper, ui) = (root, circle))
        -root > lower && ((lower, li) = (-root, -circle))
        if abs(ny) >= 1.0e-14hypot(nx, ny) && ny != 0
            line = (a - nx * mid) / ny
            integral = (a * width - nx * (right * right - left * left) / 2) / ny
            ny > 0 && line < upper && ((upper, ui) = (line, integral))
            ny < 0 && line > lower && ((lower, li) = (line, integral))
        end
        upper > lower && (area += ui - li)
    end
    return max(0.0, area)
end

function circle_symmetric_difference(f, reference; center = (0.5, 0.75), radius = 0.15)
    mesh = f.mesh;difference = 0.0
    reconstruct!(f)
    for ref in TwoPhaseAMR._cell_references(mesh)
        c = TwoPhaseAMR._get(f.fraction, ref)
        exact = TwoPhaseAMR._get(reference.fraction, ref)
        block = mesh.blocks[ref.block];volume = block.volume
        if c == 0 || exact == 0
            difference += volume * (c + exact)
        elseif c == 1 || exact == 1
            difference += volume * abs(c - exact)
        else
            x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
            dx, dy = block.spacing;x -= center[1];y -= center[2]
            nx = TwoPhaseAMR._get(f.normal_x, ref) / dx
            ny = TwoPhaseAMR._get(f.normal_y, ref) / dy
            a = TwoPhaseAMR._get(f.intercept, ref) + nx * x + ny * y
            overlap = circle_plane_overlap(x - dx / 2, x + dx / 2, y - dy / 2, y + dy / 2, nx, ny, a; radius)
            difference += max(0.0, volume * (c + exact) - 2overlap)
        end
    end
    return difference
end
