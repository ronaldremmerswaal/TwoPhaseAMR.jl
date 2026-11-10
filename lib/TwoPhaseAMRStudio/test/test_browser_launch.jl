using Test
using TwoPhaseAMR
using TwoPhaseAMRStudio

@testset "Browser launch keeps credentials out of process arguments" begin
    if Sys.islinux()
        mktempdir() do dir
            opener = joinpath(dir, "xdg-open")
            capture = joinpath(dir, "argument")
            write(opener, "#!/bin/sh\nprintf '%s' \"\$1\" > \"\$STUDIO_TEST_CAPTURE\"\n")
            chmod(opener, 0o700)
            oldpath = get(ENV, "PATH", nothing)
            oldcapture = get(ENV, "STUDIO_TEST_CAPTURE", nothing)
            app = nothing
            try
                ENV["PATH"] = dir
                ENV["STUDIO_TEST_CAPTURE"] = capture
                app = studio(Recorder(); open = :browser)
                @test timedwait(() -> isfile(capture) && filesize(capture) > 0, 30.0; pollint = 0.01) === :ok
                argument = read(capture, String)
                @test isfile(argument)
                @test (stat(dirname(argument)).mode & 0o077) == 0
                @test (stat(argument).mode & 0o077) == 0
                @test !occursin(app.token, argument)
                @test occursin(url(app), read(argument, String))
                close(app)
                @test !ispath(argument)
            finally
                isnothing(app) || close(app)
                isnothing(oldpath) ? delete!(ENV, "PATH") : (ENV["PATH"] = oldpath)
                isnothing(oldcapture) ? delete!(ENV, "STUDIO_TEST_CAPTURE") : (ENV["STUDIO_TEST_CAPTURE"] = oldcapture)
            end
        end
    end
end
