# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from transformers import HubertForCTC, HubertConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from tqdm import tqdm

class MyDataset(data.Dataset):

    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class MyCollator:

    def __init__(self, args, phones):
        self.args = args
        self.phones = phones
        self.tokenizer = Wav2Vec2CTCTokenizer(
            args.token,
            vocabsize=len(phones),
            unk_token="?",
            pad_token="_",
            word_delimiter_token="|",
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

    def __call__(self, batch):
        batch, batch_label = list(zip(*batch))
        max_len = 0
        max_label = 0
        for item in batch:
            max_len = max(max_len, len(item["audio"]["array"]))
            max_label = max(max_label, len(item["label"]))
        batch_list = [[], [], []]  # input_values, attention_mask, label
        for i in range(len(batch)):
            audio_data = self.processor(batch[i]["audio"]["array"],
                                        sampling_rate=16000,
                                        return_tensors="pt",
                                        padding="max_length",
                                        max_length=max_len,
                                        pad_to_max_length=True)

            batch_list[0].append(audio_data["input_values"])
            batch_list[1].append(audio_data["attention_mask"])
            batch_list[2].append(batch_label[i][:max_label])
        batch_data = {
            "input_values": torch.stack(batch_list[0]),
            "attention_mask": torch.stack(batch_list[1]),
            "labels": torch.stack(batch_list[2])
        }
        return batch_data

# RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 23.69 GiB total capacity; 21.28 GiB already allocated; 137.25 MiB free; 21.56 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
