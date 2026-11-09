using Test

@testset "One-thread Studio fallback" begin
    if Threads.nthreads(:default) == 1 && Threads.nthreads(:interactive) == 0
        include("single_thread_fixture.jl")
    else
        # Exercise the actual fallback even when Pkg.test has interactive threads.
        # Keep the Pkg.test sandbox so the child uses exactly the resolved package
        # and test dependencies, but force a fresh process with one Julia thread.
        project = dirname(Base.active_project())
        fixture = joinpath(@__DIR__, "single_thread_fixture.jl")
        # Julia 1.11 rejects an explicit zero-sized interactive pool; `-t 1`
        # already has no interactive threads there. Julia 1.12 defaults to one.
        threads = VERSION < v"1.12" ? "--threads=1" : "--threads=1,0"
        command = `$(Base.julia_cmd()) --startup-file=no --heap-size-hint=3G $threads --project=$project $fixture`
        mktempdir() do directory
            open(joinpath(directory, "child.log"), "w+") do log
                process = run(pipeline(ignorestatus(command), stdout = log, stderr = log); wait = false)
                status = timedwait(() -> Base.process_exited(process), 420.0; pollint = 0.05)
                if status !== :ok
                    kill(process)
                end
                wait(process)
                flush(log)
                seekstart(log)
                output = read(log, String)
                if status !== :ok || !success(process)
                    @error "Single-thread fixture failed or timed out" output
                end
                @test status === :ok
                @test success(process)
            end
        end
    end
end
