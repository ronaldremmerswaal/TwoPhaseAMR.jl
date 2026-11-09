@assert startswith(Base.active_project(), normpath(joinpath(@__DIR__, "..", "..")))

using LinearAlgebra
BLAS.set_num_threads(1)
using JSON
using TwoPhaseAMR

function camera_vectors()
    inputs = (
        ((0.0, 1.0, 0.0, 1.0), (800, 400), 0.5, (250.0, 110.0)),
        ((-3.0, 2.0, 4.0, 6.0), (300, 600), 1.8, (120.0, 500.0)),
        ((0.2, 0.8, -1.0, 2.0), (640, 360), 0.25, (0.0, 0.0)),
        ((-10.0, 5.0, -4.0, 9.0), (1024, 768), 2.0, (1024.0, 768.0)),
    )
    return map(inputs) do (rect, size, factor, anchor)
        fit = fit_camera(rect, size)
        # The browser uses a scale multiplier and top-left origin; Julia's
        # zoom uses its reciprocal and a lower-left pixel origin.
        zoomed = zoom(fit, inv(factor), (anchor[1], size[2] - anchor[2]), size)
        return (;
            rect, size, fit = (; center = fit.center, scale = fit.scale),
            visible = visible_rect(fit, size),
            zoom = (; factor, anchor, center = zoomed.center, scale = zoomed.scale),
        )
    end
end

open(joinpath(@__DIR__, "camera-vectors.json"), "w") do io
    JSON.print(io, camera_vectors())
    println(io)
end
