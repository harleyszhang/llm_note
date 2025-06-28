import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

comm.Barrier()

N = 5
if comm.rank == 0:
    A = np.arange(N, dtype=np.float64)    # rank 0 初始化数据到变量 A
else:
    A = np.empty(N, dtype=np.float64)     # 其他节点的变量 A 为空

# 广播
comm.Bcast([A, MPI.DOUBLE])

# 验证所有节点上的 A
print("Rank:%2d, data:%s" % (comm.rank, A))