"""
    Camera(center, scale)

Isotropic two-dimensional view: `center` in world coordinates and `scale` in
world units per pixel, equal in x and y. A `w × h` pixel panel shows
`center ± (w, h) .* scale ./ 2`, so the view always fills the panel and cells
stay square. Pixel coordinates are measured from the panel's lower-left
corner. See [`fit_camera`](@ref), [`zoom`](@ref), [`zoom_to_box`](@ref), and
[`visible_rect`](@ref).
"""
struct Camera
    center::NTuple{2, Float64}
    scale::Float64

    function Camera(center, scale::Real)
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("the camera scale must be finite and positive"))
        all(isfinite, center) ||
            throw(ArgumentError("the camera centre must be finite"))
        return new((Float64(center[1]), Float64(center[2])), Float64(scale))
    end
end

function _check_panel(size)
    w, h = size
    w > 0 && h > 0 || throw(ArgumentError("the panel size must be positive"))
    return Float64(w), Float64(h)
end

"""
    visible_rect(camera, (w, h)) -> (xmin, xmax, ymin, ymax)

World rectangle shown by a `w × h` pixel panel.
"""
function visible_rect(camera::Camera, size)
    w, h = _check_panel(size)
    half_width = w * camera.scale / 2
    half_height = h * camera.scale / 2
    return (
        camera.center[1] - half_width, camera.center[1] + half_width,
        camera.center[2] - half_height, camera.center[2] + half_height,
    )
end

_domain_rect(rect::NTuple{4, Real}) = Float64.(rect)
_domain_rect(topology::TopologySnapshot) = (
    Float64(topology.origin[1]), Float64(topology.origin[1] + topology.extent[1]),
    Float64(topology.origin[2]), Float64(topology.origin[2] + topology.extent[2]),
)
_domain_rect(frame::Frame) = _domain_rect(frame.topology)
_domain_rect(mesh::BlockTreeMesh) = (
    Float64(mesh.origin[1]), Float64(mesh.origin[1] + mesh.extent[1]),
    Float64(mesh.origin[2]), Float64(mesh.origin[2] + mesh.extent[2]),
)

"""
    zoom_to_box(camera, (xmin, xmax, ymin, ymax), (w, h))

Centre the box and choose the smallest isotropic scale that shows all of it;
the extra extent of the panel shows more of the domain. A degenerate box
returns `camera` unchanged.
"""
function zoom_to_box(camera::Camera, box, size)
    w, h = _check_panel(size)
    xmin, xmax, ymin, ymax = Float64.(box)
    width, height = xmax - xmin, ymax - ymin
    width > 0 && height > 0 || return camera
    return Camera(((xmin + xmax) / 2, (ymin + ymax) / 2), max(width / w, height / h))
end

"""
    fit_camera(domain, (w, h))

Camera showing the whole `domain` (a frame, topology snapshot, mesh, or
`(xmin, xmax, ymin, ymax)` tuple) in a `w × h` pixel panel. The panel is
filled along the limiting direction and the domain is centred.
"""
fit_camera(domain, size) =
    zoom_to_box(Camera((0.0, 0.0), 1.0), _domain_rect(domain), size)

"""
    zoom(camera, factor, anchor, (w, h))

Zoom by `factor` (> 1 zooms in) while keeping the world point under pixel
`anchor = (px, py)` fixed.
"""
function zoom(camera::Camera, factor::Real, anchor, size)
    isfinite(factor) && factor > 0 ||
        throw(ArgumentError("the zoom factor must be finite and positive"))
    xmin, _, ymin, _ = visible_rect(camera, size)
    w, h = _check_panel(size)
    px, py = Float64(anchor[1]), Float64(anchor[2])
    world = (xmin + px * camera.scale, ymin + py * camera.scale)
    scale = camera.scale / factor
    return Camera(
        (world[1] + (w / 2 - px) * scale, world[2] + (h / 2 - py) * scale), scale,
    )
end
