#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char route_buf1[512];
static char route_buf2[512];
static char route_buf3[512];
static char exec_local[512];

static char* build_prefix(char* in) {
    snprintf(route_buf1, sizeof(route_buf1), "runner %s", in);
    return route_buf1;
}

static char* add_route_flags(char* in) {
    snprintf(route_buf2, sizeof(route_buf2), "%s --channel=cli --retry=0", in);
    return route_buf2;
}

static char* finalize_route(char* in) {
    snprintf(route_buf3, sizeof(route_buf3), "%s --complete", in);
    return route_buf3;
}

char* stage_route(char* in) {
    char* a = build_prefix(in);
    char* b = add_route_flags(a);
    char* c = finalize_route(b);
    return c;
}

void stage_execute(char* cmd) {
    size_t n;

    if (cmd == NULL) {
        return;
    }
    n = strlen(cmd);
    if (n >= sizeof(exec_local)) {
        n = sizeof(exec_local) - 1;               
    }


    exec_local[n] = '\0';                              
    system(exec_local); /* external command execution */
}
