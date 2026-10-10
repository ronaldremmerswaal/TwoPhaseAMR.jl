/* Optional per-thread counters around complete Fortran toy trajectories.
 * Linux generic events are not a direct DRAM-byte or L3-miss measurement.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

enum { NUM_EVENTS = 5 };
static __thread int fds[NUM_EVENTS] = {-1, -1, -1, -1, -1};
static const uint32_t types[NUM_EVENTS] = {
    PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE,
    PERF_TYPE_RAW, PERF_TYPE_RAW,
};
static const uint64_t generic_configs[3] = {
    PERF_COUNT_HW_CPU_CYCLES,
    PERF_COUNT_HW_INSTRUCTIONS,
    PERF_COUNT_HW_CACHE_MISSES,
};
static const uint64_t origin_configs[2][2] = {
    {0x0843, 0x0243}, /* Zen3 demand fill, local memory/IO vs same-CCX cache. */
    {0x085a, 0x025a}, /* Zen3 hardware-prefetch fill, same two origins. */
};

static int event_open(struct perf_event_attr *attr) {
    return (int)syscall(__NR_perf_event_open, attr, 0, -1, -1, 0);
}

int counter_open_thread(int mode) {
    if (mode != 1 && mode != 2) return -EINVAL;
    for (int e = 0; e < NUM_EVENTS; ++e) {
        struct perf_event_attr attr = {0};
        attr.type = types[e];
        attr.size = sizeof(attr);
        attr.config = e < 3 ? generic_configs[e] : origin_configs[mode - 1][e - 3];
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        fds[e] = event_open(&attr);
        if (fds[e] < 0) {
            int saved = errno;
            for (int k = 0; k < e; ++k) { close(fds[k]); fds[k] = -1; }
            return -saved;
        }
    }
    return 0;
}

int counter_start_thread(void) {
    for (int e = 0; e < NUM_EVENTS; ++e) {
        if (fds[e] < 0) return -EBADF;
        if (ioctl(fds[e], PERF_EVENT_IOC_RESET, 0) < 0) return -errno;
        if (ioctl(fds[e], PERF_EVENT_IOC_ENABLE, 0) < 0) return -errno;
    }
    return 0;
}

int counter_stop_thread(int64_t result[NUM_EVENTS * 3]) {
    for (int e = 0; e < NUM_EVENTS; ++e) {
        uint64_t values[3] = {0};
        if (fds[e] < 0) return -EBADF;
        if (ioctl(fds[e], PERF_EVENT_IOC_DISABLE, 0) < 0) return -errno;
        ssize_t nread = read(fds[e], values, sizeof(values));
        if (nread != (ssize_t)sizeof(values)) return nread < 0 ? -errno : -EIO;
        for (int k = 0; k < 3; ++k) result[3 * e + k] = (int64_t)values[k];
        close(fds[e]);
        fds[e] = -1;
    }
    return 0;
}
