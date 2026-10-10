# Stage 3 large-mesh construction crash: investigation in progress, 10 October 2026

The ordered-gather copied-source experiment has **no accepted timing rows**.
Its unchanged control crashed while constructing a 2048² periodic one-block
mesh, before advection. A fresh 2048² construction also crashed without any
preceding 1024² case. Later checked scans found invalid `InteriorFace` fields
*immediately after face-schedule construction*, before halo construction or
dual-volume calculation. This locates the first observed bad values but does
not establish whether they came from face production, earlier memory damage,
or a runtime/compiler problem. The narrow-ID 90-row cohort completed under
the same frozen production source; its accepted timings remain separate from
this investigation.

## Observations

| Diagnostic | Recorded outcome | What it establishes |
|:--|:--|:--|
| Original ordered-gather pilot and one identical retry | Exit −11 in unchanged `mesh.jl::_face_dual_volumes` at line 208, with zero accepted timing rows in either attempt. | Failure occurs during control mesh construction, before a gather comparison. |
| Clean mesh-only, fresh 2048² and 1024²→2048² | Both exited on signal 11 at the same dual-volume line. The sequential run completed 1024² and intercase GC first. | A preceding long advection trajectory is unnecessary for a crash. |
| Fresh 2048² with `--check-bounds=yes` | Completed construction of 8,388,608 faces, with no `BoundsError`. | A checked run can succeed; one success does not establish that the unchecked code or hardware is sound. |
| Copied-source constructor hoist | Both original and hoisted variants exited −11. | This one liveness/codegen change did not remove the failure. It is not a timing comparison. |
| First staged construction | One process completed root, one leaf block, 8,388,608 faces, 8,196 halo entries, and a full face-range scan. It invoked the original dual-volume helper but consumed only `length(dual)`. | This process's face scan passed. The helper's arithmetic and stores may have been optimized away, so this is not proof of a complete successful dual-volume calculation. Face count alone would be a weaker check and cannot establish stored-reference validity. |
| Second staged construction | Checked validation stopped before dual volumes at face 237572: `minus=CellRef(1,1185,4)`, `plus=CellRef(262145,1186,4)`, axis 1. It exited 1 intentionally, not by signal. | With one block, `plus.block=262145` is invalid. Its difference from expected block 1 is bit 18; the bit pattern does not identify a cause. |
| Two fresh phase-scan processes | Found 226 and 263 invalid faces respectively **immediately after** `_build_face_schedules`, before `_build_halo_schedule`. Each kept the same invalid count after halo; sampled bad fields were stable on immediate and later rereads. In each process, the all-field sum increased by `0x80000` between scans and the FNV checksum changed. Dual volumes were skipped. | Invalid fields precede halo and the faulting dual call in these runs. Equal counts do not imply identical arrays or prove that halo had no effect. |
| Matched original-source fresh 2048² under Julia 1.12.7 and 1.10.12 | Both used the same copied validation driver and frozen CFD source/project; each returned zero bad faces and zero bad dual values across 8,388,608 entries. The Julia 1.12 launch spent time precompiling and is not a timing reference. | Neither runtime failed in this one-process pair. The earlier Julia 1.10 failures are intermittent; this pair does not establish a simple version-specific cause. |
| Exact archived count-only driver, three fresh controls | With the same original driver SHA-256 `75ae5c1c…`, frozen source/project and p1 placement, Julia 1.10 default, Julia 1.10 `--inline=no`, and Julia 1.12 default each completed the 8,388,608-face mesh. The driver did **not** scan face or dual values. | These three count-only completions do not contradict the earlier Julia 1.10 default signal-11 failure or establish an inlining/runtime cause. Their elapsed times are diagnostic launch durations, not optimization measurements. |
| Separate 1 GiB `UInt64` buffer check | Julia 1.10.12 and 1.12.7 each completed one process with patterns `1` and `0xAAAAAAAAAAAAAAAA`; all four full readbacks had zero mismatches and matching checksums. | These simple buffer passes did not reproduce corruption; they do not exercise face construction and are not a hardware certification. |
| Standalone 80-byte tuple-vector growth | One Julia 1.10.12 process appended 8,388,608 `NTuple{10,UInt64}` entries without a size hint, then safely read all 83,886,080 deterministic words; zero mismatches. Nominal payload was 671,088,640 B (640 MiB); observed process `VmHWM` was 1,021,840 KiB. | This simple isbits-vector growth/readback passed. It does not exercise `RawSide`, sorting, face matching, or nested face types; growth has no guaranteed 1 GiB RSS cap. |
| Corrected pre-match `RawSide` checkpoint | One copied-source Julia 1.10.12 process scanned 16,777,216 entries, exactly 4,194,304 in each of four axis/polarity groups, hashed all direct backing words, and checked every record's fixture invariants through typed fields. It found zero invalid records under those checks and then intentionally stopped with exit 1 before face matching. | The produced `RawSide` tables satisfied these invariants in this process at that checkpoint. This does **not** establish that the earlier processes with bad final faces had valid `RawSide` inputs. |
| Same-process complete face-schedule checkpoints | In each of two fresh copied-source Julia 1.10.12 processes, `RawSide` checks passed, the unchanged vertical and horizontal matchers ran, and scans found zero invalid x faces before/after horizontal matching and zero invalid y faces. Each axis had 4,194,304 faces. The x-prefix FNV and sum agreed before/after horizontal append in each process; both intentionally exited 1 before halo/dual work. | These two instrumented full-schedule runs passed their fixture checks, and the checksum agreement does not prove byte-for-byte identity. Earlier uninstrumented bad-face runs remain unexplained; the observations do not identify or clear the horizontal matcher as a general cause. |

The phase-scan records show that bad values can affect several `Int` fields
within nearby faces, often adding high bits while low bits retain plausible
coordinates. For example, one immediate post-builder record has
`plus.j=0x4a0367` and `axis=0x420001`; another has
`plus.block=0xe80001`. This is stronger evidence than a face-count check,
but neither a single-bit difference nor clustered upper-bit changes prove a
hardware, GC, or compiler cause. `_build_halo_schedule` does not receive the
`faces` vector, and the observed invalid values already precede its call.
The unchecked dual-volume indexing provides a plausible path from an invalid
reference to the signal-11 stack, without proving that every earlier crash
had the same immediate bad face.

The first `RawSide` checkpoint is **excluded as a harness failure**: its
opposite-polarity assumption marked all 16,777,216 entries invalid despite
plausible direct backing words. The corrected polarity/wrap rule passed a
synthetic 2×2 valid case and uniquely rejected an injected bad block ID
`262145` before the large checkpoint above. Those checks validate this
probe's interpretation; they do not retroactively validate the `RawSide`
state in the two bad-face phase-scan processes.

Unprivileged kernel-log inspection gave no usable fault telemetry:
`journalctl -k` returned no warning-or-higher entries in the queried window,
`dmesg` denied access, and no EDAC memory-controller counters were exposed.
These access limits cannot be read as evidence for or against a hardware
fault. No hardware cause is asserted.
At a later read-only check, the parent environment had none of the queried
`LD_PRELOAD`, `LD_LIBRARY_PATH`, `JULIA_CPU_TARGET`,
`JULIA_GC_ALLOC_POOL`, `JULIA_GC_ALLOC_OTHER`, `JULIA_GC_ALLOC_PRINT`,
`JULIA_NUM_GC_THREADS`, or `MALLOC_*`
overrides. The CFD launchers explicitly set one Julia/GC thread and use
`--startup-file=no`. This current environment check does not reconstruct
every historical child environment or eliminate allocator/runtime causes.

## Frozen inputs and local receipts

The CFD diagnostics used the narrow-ID final cohort's copied production
source, whose tree SHA-256 is
`0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2`.
The mesh source SHA-256 is
`d62cebb9cb5775f3b72de974ec55e2d4bf572f6daefdb28b1a89ff46eccf86fd`;
the advection source SHA-256 is
`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`.
The primary crash diagnostics used `/tmp/julia-1.10.12/bin/julia` (binary SHA-256
`1c7ea3ba2e49e6c4a96ca8b80e34f984f65c1dbcdc1423f33a63f2bc14a69a5b`),
at one Julia thread and one GC thread, pinned to CPU 0. Its frozen numerical
Project and Manifest SHA-256 values are
`01d49519e37d146088a17f8830bff96f2e941e07c1f79ea895c2df199f1f1e91`
and `d2bfdb90a95191cd2e7ed74a32cc827e3413f05ef231de44edac5260e92f90d0`.
The explicit paired-runtime comparison also used Julia 1.12.7 (binary
SHA-256 `582718c20c563d824b88a08e2a9d0afcf5015b3053f425697cecc79ec4e9ea36`)
with the same copied CFD source, driver, and pinned numerical Project/Manifest.
The CFD source corresponds to the checked narrow-ID candidate committed in
`7824f089ea0ff8d7dfa579b999e2e8c0d4bb9bec` (also present at
`2e8195bbcb61d9fd1abe8b9272d2d4bd1cb60c81`), **not** the older
`5f0081d` block-interior control. The narrow-ID schema-2 reconstruction
builds that older control and the pinned environment; reproducing these
mesh diagnostics additionally requires the later candidate `src` tree or
the retained `current-source` snapshot with the hashes above.
Each local `configuration.json` locks source, executable, project, script,
and affinity; each `receipt*.json` records the exact child command and exit.

The Julia diagnostic launchers are
[mesh-only](../../benchmark/scaling/stage3/mesh_crash/run.jl),
[forced-bounds](../../benchmark/scaling/stage3/mesh_crash/run_checked.jl),
[constructor hoist](../../benchmark/scaling/stage3/mesh_crash/run_constructor_probe.jl),
and [staged/phase scans](../../benchmark/scaling/stage3/mesh_crash/run_staged.jl).
Use `--help` for each CLI and a **new** output directory; the copied control
source and pinned numerical environment are local frozen inputs. The
clean/checked directories retain the exact original `mesh_driver-clean-v1.jl`
(SHA-256 `75ae5c1c3f7917caa06d9eb62a6e9a1a237f300af173f113550d7db4d63ff05b`),
and staged1 retains exact `run_staged-v1.jl` and `staged_driver-v1.jl`
snapshots matching its configuration hashes. The later staged2 and
phase-scan directories also retain their copied drivers. These exact early
driver snapshots are ignored local evidence, so a clean checkout needs them
or an independently reconstructed byte-matching copy for exact replay. The
[narrow-ID reconstruction recipe](../../benchmark/scaling/stage3/narrow_ids/README.md)
documents how to recreate the committed control environment for a clean
checkout. The ignored local evidence directories are:

- [`20261010-ordered-gather-pilot/`](../../benchmark/scaling/results/20261010-ordered-gather-pilot/) and [`...-retry1/`](../../benchmark/scaling/results/20261010-ordered-gather-pilot-retry1/) for the original signal-11 failures;
- [`20261010-mesh-crash-clean1/`](../../benchmark/scaling/results/20261010-mesh-crash-clean1/), [`...-checked1/`](../../benchmark/scaling/results/20261010-mesh-crash-checked1/), and [`...-constructor-pair1b/`](../../benchmark/scaling/results/20261010-mesh-crash-constructor-pair1b/) for the clean, forced-bounds, and hoist comparisons;
- [`20261010-mesh-crash-staged1/`](../../benchmark/scaling/results/20261010-mesh-crash-staged1/), [`...-staged2/`](../../benchmark/scaling/results/20261010-mesh-crash-staged2/), and [`...-phase-scan1/`](../../benchmark/scaling/results/20261010-mesh-crash-phase-scan1/) for the ordered construction phases, invalid-face fields, checksums, and rereads;
- [`20261010-mesh-crash-runtime112-fresh1/`](../../benchmark/scaling/results/20261010-mesh-crash-runtime112-fresh1/) and [`...-runtime110-matched1/`](../../benchmark/scaling/results/20261010-mesh-crash-runtime110-matched1/) for the matched copied driver (SHA-256 `24c6ac0599d4f21b30a8e0417222cdc90268dafb6b8b5091f0e083acf1055077`) and full face/dual validation; and [`...-face-producer1/`](../../benchmark/scaling/results/20261010-mesh-crash-face-producer1/) for the ten-minute timeout receipt;
- [`20261010-mesh-crash-clean-controls1/`](../../benchmark/scaling/results/20261010-mesh-crash-clean-controls1/) for the three exact [archived count-only driver](../../benchmark/scaling/stage3/mesh_crash/mesh_driver_clean_v1.jl) launches, with commands, runtime hashes, events, and exits;
- [`20261010-memory-integrity-julia110/`](../../benchmark/scaling/results/20261010-memory-integrity-julia110/) and [`...-julia112/`](../../benchmark/scaling/results/20261010-memory-integrity-julia112/) for the separate [buffer-check script](../../benchmark/scaling/stage3/mesh_crash/memory_integrity.jl), SHA-256 `61799e393d055924e186b9680367d8508a0cb976cd6e6bea5ae798593a86dd93`.
- [`20261010-mesh-crash-tuple-growth110/`](../../benchmark/scaling/results/20261010-mesh-crash-tuple-growth110/) for the standalone [tuple-growth script](../../benchmark/scaling/stage3/mesh_crash/tuple_growth_integrity.jl), SHA-256 `c20ff4cd75c06d6986da34b25be763ce669106932301b3c767d803c2d8c62ee0`, and its checked full-field receipt.
- [`20261010-mesh-crash-raw-checkpoint1/`](../../benchmark/scaling/results/20261010-mesh-crash-raw-checkpoint1/) for the excluded opposite-polarity harness failure, and [`...-corrected1/`](../../benchmark/scaling/results/20261010-mesh-crash-raw-checkpoint-corrected1/) for the corrected `RawSide` backing-word receipt. The [checkpoint generator](../../benchmark/scaling/stage3/mesh_crash/make_raw_checkpoint.jl) and [synthetic polarity/bad-block check](../../benchmark/scaling/stage3/mesh_crash/check_raw_checkpoint.jl) are Julia-only. The corrected copied mesh SHA-256 is `29d18ff83d075cd7aa83eaaecf51e3c83ff38fea36a6290975a27efb576c3410` and its patch SHA-256 is `7a505d23af8fbc5ea8f4457c2e5f58ef7efc93d02483818d108a2a7db7551193`.
- [`20261010-mesh-crash-vertical-checkpoint1/`](../../benchmark/scaling/results/20261010-mesh-crash-vertical-checkpoint1/) for the two same-process `RawSide`→vertical-face receipts. Its [generator](../../benchmark/scaling/stage3/mesh_crash/make_vertical_checkpoint.jl) produced copied mesh SHA-256 `43c84ce60dfd1195aeac389293ceb517f8a547c378ce60db4e445ea7e18fe74e` and patch SHA-256 `c229386741b170f1d257e5546db95f500720cfbb24aa5231fc3685e95652201a`; both copied drivers and configuration are retained locally.
- [`20261010-mesh-crash-full-face-checkpoint1/`](../../benchmark/scaling/results/20261010-mesh-crash-full-face-checkpoint1/) for the two complete face-schedule checkpoint receipts, including `raw-r*.toml`, `x-before`/`x-after` and y-face data, and intentional pre-halo stops. Its [generator](../../benchmark/scaling/stage3/mesh_crash/make_full_face_checkpoint.jl) produced copied mesh SHA-256 `451806906a81ab0e2ea6e4a43758d3cdb03f984cffaa9e5837b44d68efd15a77` and patch SHA-256 `c3dde2995688a760910abdb73cadae962189ea27842d3f6ea8dc6d053938b632`.

All listed result directories are ignored local evidence, not versioned
release assets. The buffer check used only Julia standard libraries,
`--check-bounds=yes`, and each runtime's default active environment; it did
not use the CFD numerical project. Its one 1 GiB allocation and two pattern
passes per process are bounded diagnostics, not stress testing or solver
measurements.

The tuple-growth check likewise used Julia standard libraries and the
default Julia 1.10.12 environment, with one thread and
`--check-bounds=yes`. Its deterministic 80-byte entries and ordinary
`push!` growth test one narrower storage mechanism, not the full mesh
producer. Its observed high-water memory is a receipt value, not a
promised cap or a throughput measurement.

The first bounded face-producer probe reached its 10-minute cap and its
wrapper was interrupted with exit 130. Its last recorded phases were
`module_loaded` and `before_mesh`; it produced no face-validation result.
That timeout is inconclusive about the producer and must not be counted as
either a pass or a detected bad face. The subsequent matched Julia 1.12.7
and 1.10.12 full-validation runs both passed, leaving the mechanism open.
The two same-process complete face-schedule checkpoints pass but stop before
halo and dual-volume work. The subsequent exact-driver three-way control
also completes, but checks only mesh/face count. Neither erases the earlier
invalid-face and signal-11 receipts. The failure remains intermittent and
unlocalized within the original large-mesh construction path. Keep copied
instrumentation outside the accepted narrow-ID implementation. The
recommendation at this point was to defer ordered-gather throughput
comparisons until numerical setup could be validated, and never to mix a
changed mesh builder into a kernel timing claim.

The subsequent [ordered-gather follow-up](multithreading-stage3-ordered-gather-report.md)
resumes ordinary numerical optimization under the user's renewed request.
It uses fresh single-case children, scans the full face/halo schedule before
timing, and stops an entire case cohort on an invalid or incomplete receipt.
Its accepted 1024² one-/many-block fixtures are separate matched results; the
many-block helper regression led to production rejection. This procedural
gate does not identify or fix the intermittent construction failure above;
the earlier crash receipts and the absence of accepted rows in the original
pilot remain
unchanged.
Its later 2048² one-block follow-up also stopped before timing: the copied
control passed compact tests, but the full oracle mesh scan found an invalid
axis at face 6,163,788 and rejected the fixture with exit 1. This is further
evidence that the construction failure remains unresolved, with zero accepted
2048² ordered-gather timing rows; it is not a new kernel result.
