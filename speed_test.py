from mapper import sum_test
import numpy as np
import time
from shared_memory_wrapper import to_shared_memory, from_shared_memory, SingleSharedArray, free_memory

<<<<<<< HEAD
numbers = np.random.randint(0, 100, 100000000, dtype=np.uint64)

to_shared_memory(SingleSharedArray(numbers), "numbers")
numbers2 = from_shared_memory(SingleSharedArray, "numbers").array

t = time.perf_counter()
print(sum_test(numbers))
print(time.perf_counter()-t)

t = time.perf_counter()
print(sum_test(numbers2))
print(time.perf_counter()-t)

t = time.perf_counter()
print(np.sum(numbers))
print(time.perf_counter()-t)

t = time.perf_counter()
print(np.sum(numbers2))
print(time.perf_counter()-t)


t = time.perf_counter()
print(sum_test(numbers2))
print(time.perf_counter()-t)

t = time.perf_counter()
print(sum_test(numbers))
print(time.perf_counter()-t)

=======
numbers = np.random.randint(0, 100000, 100000000, dtype=np.int64)
b = np.random.randint(0, 100, 1000000, dtype=np.int32)

to_shared_memory(SingleSharedArray(numbers), "numbers")
to_shared_memory(SingleSharedArray(b), "b")
numbers2 = from_shared_memory(SingleSharedArray, "numbers").array
b2 = from_shared_memory(SingleSharedArray, "b").array

t = time.perf_counter()
print(sum_test(numbers, b))
print(time.perf_counter()-t)

t = time.perf_counter()
print(sum_test(numbers2, b2))
print(time.perf_counter()-t)

t = time.perf_counter()
print(sum_test(numbers, b))
print(time.perf_counter()-t)

t = time.perf_counter()
print(sum_test(numbers2, b2))
print(time.perf_counter()-t)
>>>>>>> debugging-cython2
free_memory()