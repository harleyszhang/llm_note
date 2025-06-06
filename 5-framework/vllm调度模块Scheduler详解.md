

## 介绍

总结：Scheduler 调度器下维护着 BlockSpaceManager。它负责管理 BlockAllocator（实际参与分配物理块的类）。BlockAllocator 又分成 gpu 和 cpu 两种类型，分别管理这两类设备上的物理块。

## 参考资料

- [图解大模型计算加速系列：vLLM源码解析1，整体架构](https://zhuanlan.zhihu.com/p/691045737)