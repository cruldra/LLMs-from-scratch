# 优化预训练超参数

[hparam_search.py](hparam_search.py) 脚本基于[附录D：为训练循环添加高级功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)中的扩展训练函数，旨在通过网格搜索找到最优超参数。

>[!NOTE]
此脚本运行时间会很长。您可能希望减少顶部`HPARAM_GRID`字典中探索的超参数配置数量。
