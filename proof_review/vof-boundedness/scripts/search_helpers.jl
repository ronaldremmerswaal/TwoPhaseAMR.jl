# Helpers of the overlap searches / counterexamples of the audit vof-boundedness (not a check by itself).
# Included by code_04_overlap_search.jl and code_06_counterexample.jl; needs TwoPhaseAMR, VofAudit (as in those scripts).

const G = VofAudit.Docs.Geo
const SQ = G.rect_poly(-0.5, 0.5, -0.5, 0.5)

function build_mesh(kind)
    kind == :uniform && return BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (2, 2), extent = (4.0, 4.0),
        periodic = (false, false)
    )
    return BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (2, 2), extent = (4.0, 4.0), periodic = (false, false),
        max_level = 1, refine = (k, o, e) -> k == BlockKey(0, 1, 1)
    )
end

function find_cell(mesh, x, y)
    for (b, block) in pairs(mesh.blocks), j in 1:mesh.block_size[2], i in 1:mesh.block_size[1]
        c = cellcenter(mesh, b, i, j)
        abs(c[1] - x) < 1.0e-12 && abs(c[2] - y) < 1.0e-12 && return CellRef(b, i, j)
    end
    error("no cell at ($x, $y)")
end

"""Parameters of one configuration: straight wall, straight liquid interface (angles / offsets relative to the donor centre)."""
random_params(rng) = (; tw = 2pi * rand(rng), ow = 0.6 * (rand(rng) - 0.5), tl = 2pi * rand(rng), ol = 0.8 * (rand(rng) - 0.5))

function build_fields(mesh, donor_centre, p; artifact = false)
    nw = (cos(p.tw), sin(p.tw)); nl = (cos(p.tl), sin(p.tl))
    cx, cy = donor_centre
    liquid = (x, y) -> nl[1] * (x - cx) + nl[2] * (y - cy) - p.ol
    if artifact
        # Only the donor is cut: its stored solid plane is set by hand (open part {nw . xi >= ow} in local coordinates), every
        # other cell is uncut, so the neighbours' traces of the donor's faces are 1 and the package's own aperture rule gives
        # a_f = (trace_donor + 1)/2.  This is the face-trace averaging artefact of the audit of the triangle example.
        geometry = GeometryField(mesh, (x, y) -> y + 100.0; lipschitz = 1)
        donor = find_cell(mesh, cx, cy)
        open = G.clip_halfplane(SQ, nw, p.ow; keep_less = false)
        area = length(open) >= 3 ? G.shoelace(open) : 0.0
        0.02 < area < 0.98 || throw(ArgumentError("degenerate open part"))
        AMR._set!(geometry.solid.fraction, 1 - area, donor)
        AMR._set!(geometry.solid.normal_x, nw[1], donor); AMR._set!(geometry.solid.normal_y, nw[2], donor)
        AMR._set!(geometry.solid.intercept, p.ow, donor)
        AMR._set!(geometry.open_fraction, area, donor)
        for (index, face) in pairs(mesh.faces)
            geometry.aperture[index] = AMR._interior_open_aperture(geometry.solid, face)
        end
    else
        # solid where nw . (x - c) - ow <= 0
        geometry = GeometryField(mesh, (x, y) -> nw[1] * (x - cx) + nw[2] * (y - cy) - p.ow; lipschitz = 1)
    end
    field = InterfaceField(mesh, liquid; geometry, lipschitz = 1)
    reconstruct!(field)
    return field
end

"""Donor faces of the x sweep: (right segments, left faces)."""
function donor_faces(mesh, donor)
    right = [i for (i, f) in pairs(mesh.faces) if f.axis == 1 && f.minus == donor]
    left = [i for (i, f) in pairs(mesh.faces) if f.axis == 1 && f.plus == donor]
    return right, left
end
