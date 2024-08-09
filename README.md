## OmniBal

![framework](./images/framework.png)
Balanced Dynamic Mini-Batch for our paper
**[OmniBal: Towards Fast Instruct-tuning for Vision-Language Models via  Omniverse Computation Balance](https://arxiv.org/abs/2407.20761)**

### InterVL Code Example

[InternVL-Chat-V1.5](https://github.com/ModelTC/InternVL/tree/OmniBal_V1.5)
[InternVL-Chat-V2.0](https://github.com/ModelTC/InternVL/tree/OmniBal_V2.0)

TODO
- [x] Add InternVL Example
- [ ] Add InternVL Train Readme Example
- [ ] Add XTuner Example
- [ ] Add LLava Example

### How to Run ISF
![ISF](./images/data_group.png)

#### Prepare dataset length

We need to calculate offline statistics for all data, including the number of images and the token number of text.

We have already prepared the internvl-1.2M length information and placed it in the dataset.
test_balanced_dynamic_batch.py

#### Data Input

"internvl_sft_1.2M.json" is our simulated input, containing actual real statistical lengths.

The "Token_length" information consists of a list in this data format. "vit_num" represents the vision image batch size number in the current sample, "token_num" indicates the final text token length, and "image_flag" refers to the actual number of images in a sample. (Some plain text might generate fake images as dummy inputs to ensure training stability.)

```json
[
    {"vit_num": 5,
      "token_num": 811,
      "image_flag": 3
    },
    {"vit_num": 3,
      "token_num": 831,
      "image_flag": 3
    },
    {"vit_num": 1,
      "token_num": 310,
      "image_flag": 1
    },
    {"vit_num": 1,
      "token_num": 920,
      "image_flag": 0
    },
]

```

#### Get ISF arguments (vit bs num and llm token length)

```python
python test_balanced_dynamic_batch.py
```

*if you want to use fast version*

```
cd fast_isf
sh build.sh && cd ..
python test_balanced_dynamic_batch.py
```

#### Replace your dataset

The example implementation we provided is based on a fake dataset. For actual use, you need to replace it with your own dataset.

### Full Code

[Example](https://github.com/ModelTC/EasyLLM)



### Citation
If you find this repository helpful, please cite the paper below.

```bibtex
@article{yao2024omnibal,
  title={OmniBal: Towards Fast Instruct-tuning for Vision-Language Models via Omniverse Computation Balance},
  author={Yao, Yongqiang and Tan, Jingru and Hu, Jiahao and Zhang, Feizhao and Jin, Xin and Li, Bo and Gong, Ruihao and Liu, Pengfei},
  journal={arXiv e-prints},
  pages={arXiv--2407},
  year={2024}
}
```

### License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.
The content of this project itself is licensed under the [Apache license 2.0](./LICENSE).

## Acknowledgement

We build our project based on:
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed.git)
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed.git)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git)
- [Flash Attention 1&2](https://github.com/Dao-AILab/flash-attention)
- [LightLLM](https://github.com/ModelTC/lightllm)
- [Huggingface Transformers](https://github.com/huggingface/transformers.git)




