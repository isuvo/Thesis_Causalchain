#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern char* wrap_a(char* in);
extern char* wrap_b(char* in);
extern char* wrap_c(char* in);
extern void run_system(char* cmd);

int main(int argc, char** argv) {
    char buf[256];
    // ROOT: untrusted input (simulate reading from argv)
    if (argc > 1) {
        strncpy(buf, argv[1], sizeof(buf)-1);
        buf[sizeof(buf)-1] = '\0';
        char* p = wrap_a(buf);
        p = wrap_b(p);
        p = wrap_c(p);
        // final call: dangerous sink
        run_system(p);
    }
    return 0;
}
