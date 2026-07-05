#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void run_system(char* cmd) {
    // SINK: vulnerable external command execution
    // (intentionally no sanitization)
    system(cmd);
}
