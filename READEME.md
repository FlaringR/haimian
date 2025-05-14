重构原始项目 haimian 代码。 对日内股票数据进行预测。 
支持：多任务学习，时序预测， DeepFM类推荐系统模型, 表格深度学习。 


主要思路:根据pytorch lighting结构对代码进行重构。
模型和训练的管理： pytorch lighting 
标准的 Pytorch Lighting结构：
```
project/
│
├── configs/                  # 配置文件
│   └── config.yaml           # 超参数配置文件
│
├── data/                     # 数据相关代码
│   └── data_module.py        # 定义 LightningDataModule
│
├── models/                   # 模型相关代码
│   ├── base_model.py         # 定义基类 LightningModule
│   └── specific_model.py     # 定义具体模型
│
├── utils/                    # 工具函数
│   ├── logging.py            # 日志工具
│   └── helpers.py            # 其他辅助函数
│
├── train.py                  # 主训练脚本
├── requirements.txt          # 依赖包
└── README.md                 # 项目说明
```
Trainer由 pl提供， 可以不用写繁琐的训练代码。 在base_model下定制每个模型的前传，回传等信息。 
参数的管理：omega




代码逻辑： 

HaimianModel: 系统的核心模型， 负责整体的逻辑。 包括数据，模型等。 顶层类

BaseModel: 所有模型的抽象基类， 提供标准化的训练和评估。继承pl.LightningModule

Model: 各个模型的类型。 继承nn.module


自定义： 
如果需要修改模型：请遵循规范。 embedding + backbone + head。 需要把head定义为支持多任务的字典输出。 