import multiprocessing as mp

CPU = mp.cpu_count()
N_JOBS = max(12, CPU * 1 // 2) if CPU > 12 else CPU - 1
