cdef extern from "omp.h" nogil:
    void omp_set_num_threads(int n)
    int omp_get_max_threads()

def set_num_threads(int n):
    omp_set_num_threads(n)

def get_max_threads():
    return omp_get_max_threads()
