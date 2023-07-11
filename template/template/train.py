# -*- coding: utf-8 -*-

import argparse
import os
import json

import torch
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import HubertForCTC, HubertConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Config
from datasets import load_dataset
import torch.cuda.amp as amp
# from torchinfo import summary

# 自作プログラムの読込
from my_args import set_model_parameters
import my_util
import makelabel
import levenshtein
import data_manager
import line_notify
from dockhub import dockhub
from my_collater import MyCollator, MyDataset

# init
args = dockhub.args
writer = dockhub.writer
progress_logger = dockhub.logger["progress"]
result_logger = dockhub.logger["result"]

dockhub.write_machine_info()
dockhub.setup()

DEVICE = dockhub.device
best_val = float('inf')

print(args)
phones = makelabel.get_phones_csv(args.train_audio_path)
print(phones)
train_label = makelabel.get_label_csv(args.train_audio_path, phones)
valid_label = makelabel.get_label_csv(args.valid_audio_path, phones)
test_label = makelabel.get_label_csv(args.test_audio_path, phones)

dict = {phones[i]: i for i in range(len(phones))}

# write token
with open(args.token, 'w') as vocab_file:
    json.dump(dict, vocab_file, indent=4, ensure_ascii=False)

# Load dataset
progress_logger.info("Loading Dataset")
dataset = load_dataset(args.dataset, download_mode="force_redownload")

tokenizer = Wav2Vec2CTCTokenizer(
    args.token,
    vocabsize=len(phones),
    unk_token="?",
    pad_token="_",
    word_delimiter_token="|",
)

progress_logger.info("Data load complete")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                             sampling_rate=16000,
                                             do_normalize=True,
                                             return_attention_mask=True,
                                             vocab_size=len(dict))
progress_logger.info("Defining model...")

# Set model
configuration = HubertConfig()
# configuration = Wav2Vec2Config(vocab_size=len(dict),
#    ctc_loss_reduction="mean",
#    ctc_zero_infinity=True,
#    output_attentions=True,
#    output_hidden_states=True,
#    )
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = HubertForCTC.from_pretrained('facebook/hubert-base-ls960',
                                     vocab_size=len(dict),
                                     ctc_loss_reduction="mean",
                                     ctc_zero_infinity=True,
                                     output_attentions=True,
                                     output_hidden_states=True)

# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-base-960h",
#     config=configuration,
#     # vocab_size=len(dict),
#     ignore_mismatched_sizes=True,
#     # vocab_size=len(dict),
# )

if torch.cuda.device_count() > 1 and args.multi_gpu:
    model = torch.nn.DataParallel(model)
model.to(DEVICE)

train_dataset = MyDataset(dataset["train"], train_label)
valid_dataset = MyDataset(dataset["validation"], valid_label)
test_dataset = MyDataset(dataset["test"], test_label)

collate_fn = MyCollator(args, phones)

trainloader = DataLoader(train_dataset,
                         shuffle=True,
                         batch_size=args.batch_size,
                         num_workers=args.workers,
                         pin_memory=False,
                         collate_fn=collate_fn)
validloader = DataLoader(valid_dataset,
                         shuffle=True,
                         batch_size=args.batch_size,
                         num_workers=args.workers,
                         pin_memory=False,
                         collate_fn=collate_fn)
testloader = DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        pin_memory=False,
                        collate_fn=collate_fn)

# Set Loss func
progress_logger.info(f"Model :\n{model}")
# TODO: Use torch summary (info?)

set_model_parameters(model, args, progress_logger)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

progress_logger.info("Model defined")

# format outputs
def format_outputs(verbose):
    predict = [verbose[0]]
    for i in range(1, len(verbose)):
        if verbose[i] != predict[-1]:
            predict.append(verbose[i])
    predict = [l for l in predict if l != phones.index('_')]
    return predict

def train(loader, model, optimizer, scaler, epoch):
    """

    1 epoch on the training set

    Args:
        loader : pytorch loader
        model : pytorch model
        criterion : pytorch loss function
        optimizer : pytorch optimizer
        epoch (int): epoch number

    Returns:
        ctc_losses.avg (float) : avarage criterion loss

    """
    progress_logger.info(f"Training phase , epoch = {epoch}")
    data_controller = data_manager.DataManager("train", writer)
    data_num = len(loader.dataset)  # テストデータの総数
    pbar = tqdm(total=int(data_num / args.batch_size))
    model.train()
    for i, batch in enumerate(loader):
        # データをdeviceに載せる
        # inputs = processor(inputs,sample_rate=16000)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch["input_values"] = torch.squeeze(batch["input_values"], 1)
        # batch["attention_mask"] = torch.squeeze(batch["attention_mask"], 1)
        # with amp.autocast(enabled=args.amp):
        outputs = model(**batch)
        loss = outputs.loss
        # print(batch)
        if loss == float('inf'):
            print(batch["input_values"].size())
            exit()
        print(loss)
        # 結果保存用
        batch_size = batch["input_values"].size(0)
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        optimizer.step()

        # measure performance and record loss
        data_controller.update_loss(loss.item(), batch_size)
        pbar.update(1)
        # del loss
        # torch.cuda.empty_cache()
        # p.step()
    data_controller.write(epoch)
    pbar.close()
    optimizer.param_groups[0]['lr'] *= 0.5
    progress_logger.info('Epoch: {0}\t'
                         'Loss {loss.avg:.4f}\t'.format(epoch, loss=data_controller.loss_manager.loss))

def valid(loader, model, epoch):
    """

    valid function for 1 epoch
        valid train
        write result data

    Args:
        loader : pytorch dataloader
        model : pytorch model
        criterion : pytorch loss function
        epoch (int) : epoch number

    Returns:
        ctc_losses.avg (float?) : avarage criterion loss
        accs.avg (float?) : avarage accuracy

    """
    progress_logger.info("Validation phase")
    # 各値初期化
    data_controller = data_manager.DataManager("valid", writer)
    model.eval()
    data_num = len(loader.dataset)  # テストデータの総数
    pbar = tqdm(total=int(data_num / args.batch_size))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # データをdeviceに載せる
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch["input_values"] = torch.squeeze(batch["input_values"], 1)
            labels = batch["labels"].to(DEVICE)
            # batch["attention_mask"] = torch.squeeze(batch["attention_mask"], 1)
            # with amp.autocast(enabled=args.amp):
            outputs = model(**batch)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            loss = outputs.loss
            # 結果保存用
            result_text = ""
            batch_size = batch["input_values"].size(0)
            for i in range(batch_size):
                output = predicted_ids[i]
                label = batch["labels"][i]
                pred = format_outputs(output)
                output = [phones[l] for l in output]
                pred = [phones[l] for l in pred]
                label = label.int()
                label = [phones[l] for l in label if phones[l] not in "_"]
                ter = levenshtein.calculate_error(pred, label)
                data_controller.update_acc(ter, 1)
                result_text += "-"*50 + "\n\n---Output---\n" + " ".join(output) + "\n\n---Predict---\n" + " ".join(
                    pred) + "\n\n---Label---\n" + " ".join(label) + "\n\n" + "WER : " + "\n\n"
            result_logger.info(result_text)
            # measure performance and record loss
            data_controller.update_loss(loss.item(), batch_size)
            pbar.update(1)
    pbar.close()
    data_controller.write(epoch)
    with open("build/result.txt", mode='w') as f:
        f.write(result_text)
    progress_logger.info('Val loss:{loss.avg:.4f} '
                         'Acc:{Acc.avg:4f}'.format(loss=data_controller.loss_manager.loss,
                                                   Acc=data_controller.acc_manager.total_error))
    return data_controller.loss_manager.loss.avg

def test(loader, model) -> None:
    """

    test function for 1 epoch
        test train
        write result data

    Args:
        loader : pytorch dataloader
        model : pytorch model
    """
    progress_logger.info("Test phase")
    # 各値初期化
    data_controller = data_manager.DataManager("test", writer)
    model.eval()
    data_num = len(loader.dataset)  # テストデータの総 数
    pbar = tqdm(total=int(data_num / args.batch_size))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # データをdeviceに載せる
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch["input_values"] = torch.squeeze(batch["input_values"], 1)
            # batch["attention_mask"] = torch.squeeze(batch["attention_mask"], 1)
            with amp.autocast(enabled=args.amp):
                outputs = model(**batch)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            loss = outputs.loss
            # 結果保存用
            result_text = ""
            batch_size = batch["input_values"].size(0)
            for i in range(batch_size):
                output = predicted_ids[i]
                label = batch["labels"][i]
                pred = format_outputs(output)
                output = [phones[l] for l in output]
                pred = [phones[l] for l in pred]
                label = label.int()
                label = [phones[l] for l in label if phones[l] not in "_"]
                ter = levenshtein.calculate_error(pred, label)
                data_controller.update_acc(ter, 1)
                result_text += "-"*50 + "\n\n---Output---\n" + " ".join(output) + "\n\n---Predict---\n" + " ".join(
                    pred) + "\n\n---Label---\n" + " ".join(label) + "\n\n" + "WER : " + "\n\n"
            result_logger.info(result_text)
            # measure performance and record loss
            data_controller.update_loss(loss.item(), batch_size)
            pbar.update(1)
    pbar.close()
    data_controller.write(epoch)
    with open("build/result.txt", mode='w') as f:
        f.write(result_text)
    progress_logger.info('Test loss:{loss.avg:.4f} '
                         'Acc:{Acc.avg:4f}'.format(loss=data_controller.loss_manager.loss,
                                                   Acc=data_controller.acc_manager.total_error))

progress_logger.info("Evaluate untrained valid")
# 未学習時のモデルの性能の検証
# valid_result = valid(validloader, model, 0)
# Start Train
progress_logger.info(f"LOG : Train Started ...epoch {args.end_epoch}まで")
valtrack = 0
scaler = amp.GradScaler(enabled=args.amp)
for epoch in range(args.start_epoch, args.end_epoch + 1):
    train(trainloader, model, optimizer, scaler, epoch)
    valid_loss = valid(validloader, model, epoch)
    # save model
    is_best = valid_loss <= best_val  # ロスが小さくなったか
    # save model
    if is_best:
        valtrack = 0
        best_val = valid_loss
        best_epoch = epoch
        my_util.save_checkpoint(
            {  # modelの保存
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  #必須
                'best_val': best_val,
                'optimizer_state_dict': optimizer.state_dict(),  #必須
                "scaler": scaler.state_dict(),
                'valid_loss': valid_loss
            },
            args.checkpoint)
    else:
        valtrack += 1
    # set_lr_rate(valtrack,optimizer,opts)

    progress_logger.info('Validation: %f (best) - %d (valtrack)' % (best_val, valtrack))
    if args.patience <= valtrack:
        break

my_util.load_checkpoint(best_epoch, best_val, model)
test(testloader, model)

writer.close()  # close tensorboard writer
if not args.line == "":
    line_notify.send_line_message(args.line)
progress_logger.info("Finish!!")
