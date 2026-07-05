#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char tmp1[300];
char tmp2[300];
char tmp3[300];

char* wrap_a(char* in) {
    // propagation #1 (DFG + CALL/ARG2PARAM)
    snprintf(tmp1, sizeof(tmp1), "%s", in);
    return tmp1;
}

char* wrap_b(char* in) {
    // propagation #2
    strcat(tmp2, in);
    strcat(tmp2, " --verbose");
    return tmp2;
}

char* wrap_c(char* in) {
    // propagation #3
    sprintf(tmp3, "%s %s", in, "--force");
    return tmp3;
}
