#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* stage_clean(char* in);
char* stage_route(char* in);
void stage_execute(char* cmd);

int main(int argc, char** argv) {
    char buf[256];
    size_t n;

    memset(buf, 0, sizeof(buf));

    if (fgets(buf, sizeof(buf), stdin) == NULL) {      
        return 0;                                      
    }                                                  

    n = strlen(buf);                                   
    if (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {  
        buf[n - 1] = '\0';
    }

    char* s1 = stage_clean(buf);
    char* s2 = stage_route(s1);
    stage_execute(s2);
    return 0;
}
