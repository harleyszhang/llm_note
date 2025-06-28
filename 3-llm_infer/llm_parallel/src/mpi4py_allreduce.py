# collective_ops.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

def demo_allgather():
    """
    每个进程发出局部向量，所有进程都得到完整列表
    """
    local_vec = np.arange(rank*3, rank*3 + 3, dtype='i')  # eg Rank2->[6 7 8]
    gathered = comm.allgather(local_vec)
    print(f"[ALLGATHER] Rank {rank}: local={local_vec} → gathered={gathered}")

def demo_reduce():
    """
    将标量归约到 root；这里只做求和
    """
    local_val = (rank + 1) ** 2          # 1,4,9,16
    total = comm.reduce(local_val, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"\n[REDUCE] 汇总结果 (sum) = {total}")  # 1+4+9+16=30

def demo_allreduce():
    """
    所有进程都同时得到归约值；这里做 max
    """
    local_val = (rank + 1) * 2           # 2,4,6,8
    global_max = comm.allreduce(local_val, op=MPI.MAX)
    print(f"[ALLREDUCE] Rank {rank}: local={local_val}, global_max={global_max}")

if __name__ == "__main__":
    if rank == 0:
        print(f"\n=== 进程总数: {size} ===\n")

    comm.Barrier();  demo_allgather()
    comm.Barrier();  demo_reduce()
    comm.Barrier();  demo_allreduce()