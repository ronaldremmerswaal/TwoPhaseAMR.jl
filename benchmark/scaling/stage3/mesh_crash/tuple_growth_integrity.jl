#!/usr/bin/env julia

# One bounded, dependency-free append/growth and checked readback diagnostic.
# It does not reproduce the mesh algorithms or certify hardware.
using SHA
using TOML

const ENTRY_COUNT = 1 << 23
const ENTRY_TYPE = NTuple{10,UInt64}
const NOMINAL_BYTES = ENTRY_COUNT * sizeof(ENTRY_TYPE)

function output_argument(args)
    if any(arg -> arg in ("-h", "--help"), args)
        println("Usage: julia --startup-file=no --check-bounds=yes tuple_growth_integrity.jl --output NEW_DIR")
        return nothing
    end
    length(args) == 2 && args[1] == "--output" && !isempty(args[2]) ||
        error("expected --output NEW_DIR")
    output = abspath(args[2])
    ispath(output) && error("refusing existing output path: $output")
    return output
end

hex64(value::UInt64) = "0x" * string(value; base=16, pad=16)
expected(index::Int, field::Int) = (UInt64(index) << 8) | UInt64(field)

function memory_kib()
    rss, high_water = -1, -1
    if isfile("/proc/self/status")
        for line in eachline("/proc/self/status")
            startswith(line, "VmRSS:") && (rss = parse(Int, split(line)[2]))
            startswith(line, "VmHWM:") && (high_water = parse(Int, split(line)[2]))
        end
    end
    return rss, high_water
end

function event!(output, phase; details=Dict{String,Any}())
    rss, high_water = memory_kib()
    event = Dict{String,Any}(
        "phase" => phase,
        "unix_seconds" => time(),
        "monotonic_ns" => time_ns(),
        "rss_kib" => rss,
        "vmhwm_kib" => high_water,
    )
    merge!(event, details)
    open(joinpath(output, "events.toml"), "a") do io
        TOML.print(io, Dict("events" => [event]))
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("event fsync failed")
    end
end

@noinline function append_entries!(entries::Vector{ENTRY_TYPE})
    for index in 1:ENTRY_COUNT
        push!(entries, ntuple(field -> expected(index, field), Val(10)))
    end
    return entries
end

@noinline function scan_entries(entries::Vector{ENTRY_TYPE})
    mismatch_count = 0
    first_mismatches = Dict{String,Any}[]
    checksum = UInt64(0xcbf29ce484222325)
    value_sum = zero(UInt64)
    for index in 1:ENTRY_COUNT
        entry = entries[index]
        for field in 1:10
            actual = entry[field]
            correct = expected(index, field)
            checksum = (checksum ⊻ actual ⊻ UInt64(index) ⊻ UInt64(field)) *
                       UInt64(0x100000001b3)
            value_sum += actual
            if actual != correct
                mismatch_count += 1
                if length(first_mismatches) < 10
                    push!(first_mismatches, Dict{String,Any}(
                        "index" => index,
                        "field" => field,
                        "actual_hex" => hex64(actual),
                        "expected_hex" => hex64(correct),
                        "xor_hex" => hex64(actual ⊻ correct),
                    ))
                end
            end
        end
    end
    return Dict{String,Any}(
        "mismatch_count" => mismatch_count,
        "first_mismatches" => first_mismatches,
        "checksum_hex" => hex64(checksum),
        "value_sum_hex" => hex64(value_sum),
    )
end

function main(args)
    output = output_argument(args)
    output === nothing && return 0
    Threads.nthreads() == 1 || error("this diagnostic requires one Julia thread")
    Base.JLOptions().check_bounds == 1 ||
        error("launch with --check-bounds=yes")
    mkpath(dirname(output))
    mkdir(output)
    event!(output, "before_vector";
           details=Dict("entries" => ENTRY_COUNT, "nominal_bytes" => NOMINAL_BYTES))
    entries = ENTRY_TYPE[]
    event!(output, "before_append")
    append_started = time_ns()
    append_entries!(entries)
    append_seconds = (time_ns() - append_started) / 1e9
    event!(output, "after_append";
           details=Dict("length" => length(entries)))
    length(entries) == ENTRY_COUNT || error("append length mismatch")
    scan_started = time_ns()
    scan = scan_entries(entries)
    scan_seconds = (time_ns() - scan_started) / 1e9
    event!(output, "after_scan";
           details=Dict("mismatch_count" => scan["mismatch_count"],
                        "checksum_hex" => scan["checksum_hex"]))
    rss, high_water = memory_kib()
    receipt = Dict{String,Any}(
        "schema" => 1,
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "julia_version" => string(VERSION),
        "julia_executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "active_project" => something(Base.active_project(), ""),
        "threads" => Threads.nthreads(),
        "check_bounds_option" => string(Base.JLOptions().check_bounds),
        "entry_type" => string(ENTRY_TYPE),
        "entry_bytes" => sizeof(ENTRY_TYPE),
        "entries" => ENTRY_COUNT,
        "nominal_payload_bytes" => NOMINAL_BYTES,
        "append_seconds" => append_seconds,
        "scan_seconds" => scan_seconds,
        "final_rss_kib" => rss,
        "process_high_water_kib" => high_water,
        "scan" => scan,
    )
    open(joinpath(output, "receipt.toml"), "w") do io
        TOML.print(io, receipt)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("receipt fsync failed")
    end
    return scan["mismatch_count"] == 0 ? 0 : 2
end

exit(main(ARGS))
