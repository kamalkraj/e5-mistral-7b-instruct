# e5-mistral-7b-instruct


```bash
docker build -t pytorch .
```

```bash
docker run --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd):/e5-mistral-7b-instruct/ pytorch bash
```


### Prepare data

Run `prepare_dataset` to create a similarity dataset with one postive and negative pair from SNLI. 
```bash
python prepare_dataset.py
```

### Run model

set the model cache folder `export TRANSFORMERS_CACHE=.cache/`

First, run `accelerate config --config_file ds_zero3_cpu.yaml`

check the sample file for Single GPU [here](ds_zero3_cpu.yaml)

Below given parameter is taken from the paper for finetuning. 
Adjust accroding to your dataset and usecase.

```bash
accelerate launch \
    --config_file ds_zero3_cpu.yaml \
    peft_lora_embedding_semantic_search.py \
    --dataset_name similarity_dataset \
    --max_length 512 \
    --model_name_or_path intfloat/e5-mistral-7b-instruct \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --max_train_steps 1000 \
    --gradient_accumulation_steps 2048 \
    --lr_scheduler_type linear \
    --num_warmup_steps 100 \
    --output_dir trained_model \
    --use_peft
```


[loss](loss.py) function copied from here -> https://github.com/RElbers/info-nce-pytorch