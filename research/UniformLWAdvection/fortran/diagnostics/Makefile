FC = gfortran
CC = cc
FFLAGS = -O3 -march=native -fopenmp -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror
CFLAGS = -O2 -Wall -Wextra -Werror
BUILD = build
JULIA ?= julia

.PHONY: all test
all: $(BUILD)/lw_omp_padded $(BUILD)/perf_permission

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/perf_counters.o: perf_counters.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/lw_omp_padded: lw_omp_padded.f90 $(BUILD)/perf_counters.o | $(BUILD)
	$(FC) $(FFLAGS) -J $(BUILD) -o $@ lw_omp_padded.f90 $(BUILD)/perf_counters.o

$(BUILD)/perf_permission: perf_permission.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

test: all
	$(JULIA) --startup-file=no --project=../../../../benchmark/scaling test_padded.jl
