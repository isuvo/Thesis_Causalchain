#include <cstddef>
#include <cstring>

int recv(int /*sock*/, char* /*buf*/, size_t /*n*/, int /*flags*/) { return 0; }
void process(const char* /*s*/) { /* no-op */ }

class ClassB {
public:
    
    int readFromSocket(int sock, char* tmp, size_t n) {
        return recv(sock, tmp, n, 0);
    }
};

class ClassA {
public:
   
    size_t length(const char* s) { return std::strlen(s); }

    bool isSafe(std::size_t n) { return n <= 128; }

    const char* prepare(const char* s) { return s; }
};

class ClassD {
public:
  
    void copyInto(char* buf, const char* s) {
        std::strcpy(buf, s);  
    }
};

int main() {
    ClassB B; ClassA A; ClassD D;

    char buf[16];
    char tmp[256];
    int sock = 0;

    B.readFromSocket(sock, tmp, 128);   
    std::size_t n = A.length(tmp);    
    if (A.isSafe(n)) {                  
        const char* t = A.prepare(tmp);  
        D.copyInto(buf, t);              
    }

    process(buf);                      
    return 0;
}
