
from pytorch_tabular.models import AutoIntConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig, ModelConfig
from pytorch_tabular.models import BaseModel
from pytorch_tabular.models.common.layers import Embedding1dLayer
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular import TabularModel
# from data_read_new import data_read
from data_read_new import data_read

import pytorch_lightning as pl

from torch import nn
import torch.nn.functional as F



data_config = DataConfig(
        target=target_name,
        continuous_cols=num_cols,
        categorical_cols=cat_cols,

)
print('Finished data_config')
trainer_config = TrainerConfig(
    auto_lr_find=False,
    fast_dev_run=False,
    max_epochs=epochs,
    batch_size=batch_size,
    early_stopping="valid_loss",  
    early_stopping_mode="min",  
    early_stopping_min_delta=early_stopping_min_delta,  
    early_stopping_patience=early_stopping_patience,  
    checkpoints="valid_loss",  
    load_best=True,  
    auto_select_gpus=True,  

)
print('Finished trainer_config', 'epochs:', epochs, '|', 'batch_size:', batch_size)

head_config = LinearHeadConfig(
    layers="", dropout=dropout, initialization="kaiming"

).__dict__  
optimizer_config = OptimizerConfig()
# wandb


# 定义适合不平衡数据分类的指标
metrics = [
    "f1_score",             
    "auroc",                
    "average_precision",    
    "matthews_corrcoef",   
    "specificity",          
    "accuracy"              
]

metrics_params = [
    {"num_classes": num_classes, "average": "weighted"},  
    {"num_classes": num_classes},                          
    {"num_classes": num_classes, "average": "weighted"},  
    {"num_classes": num_classes},                          
    {"num_classes": num_classes},                          
    {}  
]

metrics_prob_input = [
    False,  
    True,   
    True,   
    False,  
    False,  
    False   
]


# ----------------CategoryEmbedding---------------------------
model_config = AutoIntConfig(
    task="classification",
    head="LinearHead",  
    head_config=head_config,  
    metrics=metrics,
    metrics_params=metrics_params,
    metrics_prob_input=metrics_prob_input,
    learning_rate=learning_rate,
)

experiment_config = ExperimentConfig(
    project_name='tab2024',
    run_name="autoint" ,
    exp_watch="all",
    log_target="wandb",
    log_logits=True
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config # wandb
)



tabular_model.fit(
        train=train,
        validation=validation
        )


result = tabular_model.evaluate(test)


print(result)

pred_df = tabular_model.predict(test)
print(test[target_name].value_counts())
print('--------------------------------------')
print(pred_df['prediction'].value_counts())
print(pred_df.head())