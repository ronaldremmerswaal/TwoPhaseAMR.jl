using TwoPhaseAMR

function benchmark_initialization(n)
    mesh = BlockTreeMesh(block_size = (n, n))
    field = InterfaceField(mesh)
    calls = Ref(0)
    phi = (x, y) -> begin
        calls[] += 1
        hypot(x - 0.47, y - 0.51) - 0.21
    end
    initialize!(field, phi; lipschitz = 1) # compile and warm caches
    calls[] = 0
    measurement = @timed initialize!(field, phi; lipschitz = 1)
    high_calls = calls[]
    println("n=$n high-order seconds=$(measurement.time) bytes=$(measurement.bytes) evaluations=$high_calls volume_error=$(abs(integrate(field) - pi * 0.21^2))")
    for samples in (4, 16)
        calls[] = 0
        volume = 0.0
        dx = 1 / n
        baseline = @elapsed begin
            for j in 1:n,i in 1:n,sj in 1:samples,si in 1:samples
                x = (i - 1 + (si - 0.5) / samples) * dx
                y = (j - 1 + (sj - 0.5) / samples) * dx
                volume += (phi(x, y) <= 0) * dx^2 / samples^2
            end
        end
        println("n=$n counting=$samples seconds=$baseline evaluations=$(calls[]) volume_error=$(abs(volume - pi * 0.21^2))")
    end
    return
end
for n in (16, 32, 64)
    benchmark_initialization(n)
end
