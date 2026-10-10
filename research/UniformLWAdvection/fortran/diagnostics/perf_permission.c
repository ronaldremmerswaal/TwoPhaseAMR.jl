/* Read-only permission probe for own-thread Linux hardware counters. */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static int event_open(struct perf_event_attr *attr) {
    return (int)syscall(__NR_perf_event_open, attr, 0, -1, -1, 0);
}

int main(void) {
    struct { const char *name; uint32_t type; uint64_t config; } events[] = {
        {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {"cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {"zen3_demand_mem_io_local", PERF_TYPE_RAW, 0x0843},
        {"zen3_demand_internal_cache", PERF_TYPE_RAW, 0x0243},
        {"zen3_hwprefetch_mem_io_local", PERF_TYPE_RAW, 0x085a},
        {"zen3_hwprefetch_internal_cache", PERF_TYPE_RAW, 0x025a},
    };
    unsigned int n = sizeof(events) / sizeof(events[0]);
    printf("{\"scope\":\"own_thread_user_only\",\"events\":[");
    for (unsigned int i = 0; i < n; ++i) {
        struct perf_event_attr attr = {0};
        attr.type = events[i].type;
        attr.size = sizeof(attr);
        attr.config = events[i].config;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        int fd = event_open(&attr);
        if (i) printf(",");
        if (fd < 0) {
            printf("{\"name\":\"%s\",\"available\":false,\"errno\":%d,\"error\":\"%s\"}",
                   events[i].name, errno, strerror(errno));
        } else {
            printf("{\"name\":\"%s\",\"available\":true}", events[i].name);
            close(fd);
        }
    }
    printf("]}\n");
    return 0;
}
