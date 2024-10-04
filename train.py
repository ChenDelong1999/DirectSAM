import os
import json
import argparse
import datetime
from dataclasses import dataclass

import torch
import torch.distributed as dist

from data.create_dataset import create_dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer


@dataclass
class DataCollatorForDirectSAM:

    image_processor: AutoImageProcessor
    
    def __call__(self, features):

        if type(features[0])==dict:
            image = [feature['image'] for feature in features]
            label = [feature['label'] for feature in features]
        elif type(features[0])==tuple:
            image = [feature[0] for feature in features]
            label = [feature[1] for feature in features]
        else:
            raise ValueError("Invalid input type")

        return self.image_processor(image, label, do_reduce_labels=False, return_tensors='pt')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning DirectSAM')

    parser.add_argument('--pretrained', type=str, default="chendelong/DirectSAM-1800px-0424")
    parser.add_argument('--dataset', type=str, help='Name of the training dataset')
    parser.add_argument('--input_resolution', type=int, default=1024, help='Input resolution for the image processor')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the training')
    parser.add_argument('--num_train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=-1, help='Maximum number of training steps')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps for gradient accumulation')
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='Number of workers for the dataloader')
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=4, help='Prefetch factor for the dataloader')
    parser.add_argument('--logging_steps', type=int, default=1, help='Number of steps for logging')
    parser.add_argument('--fp16', type=bool, default=True, help='Use fp16 precision')
    parser.add_argument('--thickness', type=int, default=2, help='Thickness of the boundary')

    args = parser.parse_args()
    
    dist.init_process_group(backend='nccl')

    dataset_config = json.load(open('data/dataset_configs.json'))

    dataset = create_dataset(dataset_config[args.dataset], 'train', resolution=args.input_resolution, thickness=args.thickness)
    eval_dataset = create_dataset(dataset_config[args.dataset], 'validation', resolution=args.input_resolution, thickness=args.thickness)

    model = AutoModelForSemanticSegmentation.from_pretrained(args.pretrained, num_labels=1, ignore_mismatched_sizes=True)
    image_processor = AutoImageProcessor.from_pretrained("chendelong/DirectSAM-1800px-0424", reduce_labels=True)

    image_processor.size['height'] = args.input_resolution
    image_processor.size['width'] = args.input_resolution

    data_collator = DataCollatorForDirectSAM(image_processor)
    
    if torch.distributed.get_rank() == 0:
        print(model)
        print(f"Number of parameters: {model.num_parameters()/1e6}M,  trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")
        print(dataset)

    training_args = TrainingArguments(
        output_dir=f"runs/{args.dataset}/{datetime.datetime.now().strftime('%m%d-%H%M')}-{args.input_resolution}px-from-{args.pretrained.replace('/', '_')}",
        
        learning_rate=args.learning_rate,
        warmup_steps=5000,
        lr_scheduler_type='constant_with_warmup',

        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        save_total_limit=20,
        save_steps=2000,
        save_strategy="steps",

        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=10000,

        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        push_to_hub=False,
        fp16=args.fp16,
        torch_compile=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

