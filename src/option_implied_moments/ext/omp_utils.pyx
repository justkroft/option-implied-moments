cdef extern from * nogil:
    """
    #ifdef _OPENMP
    #include <omp.h>
    #else
    static void omp_set_num_threads(int num_threads) {
        (void)num_threads;
    }
    static int omp_get_max_threads() {
        return 1;
    }
    #endif
    """
    void omp_set_num_threads(int n)
    int omp_get_max_threads()

def set_num_threads(int n):
    omp_set_num_threads(n)

def get_max_threads():
    return omp_get_max_threads()
