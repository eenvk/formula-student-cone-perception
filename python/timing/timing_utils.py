#Renzi

import time

def elapsed_ms(start_time):
    """Return elapsed time in milliseconds"""
    return (time.perf_counter() - start_time) * 1000.0