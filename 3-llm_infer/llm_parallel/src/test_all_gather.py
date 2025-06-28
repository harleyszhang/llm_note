# allgather.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 每个进程生成一个数据
data = np.array(rank + 1)  # 每个进程的数据为其 rank + 1
gathered_data = np.zeros(size, dtype=int)

# 执行 AllGather 操作
comm.Allgather(data, gathered_data)

print(f"Rank {rank} gathered data: {gathered_data}")
