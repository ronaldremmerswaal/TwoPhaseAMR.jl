FC = gfortran
FFLAGS = -O3 -march=native -fopenmp -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror
BUILD = build
JULIA ?= julia

.PHONY: all clean test
all: $(BUILD)/lw_omp

$(BUILD)/lw_omp: lw_omp.f90 | $(BUILD)
	$(FC) $(FFLAGS) -J $(BUILD) -fopt-info-vec-optimized=$(BUILD)/vectorization.txt -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

test: all
	$(JULIA) --startup-file=no --project=../../../benchmark/scaling test.jl $(BUILD)/lw_omp

clean:
	rm -rf $(BUILD)
