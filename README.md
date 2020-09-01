# fsnlu: A Fast Development Framework for Few-shot Natural language Understanding.
Developed on the event of [SMP 2020 ECDT Task1](https://mp.weixin.qq.com/s/_dE7kDw8q7FgHfGTh8DMWw) by [Changjian Liu](https://cjliux.github.io), aiming at integrating various models for few-shot natural language understanding.

## Contents

- [fsnlu: A Fast Development Framework for Few-shot Natural language Understanding.](#fsnlu-a-fast-development-framework-for-few-shot-natural-language-understanding)
  - [Contents](#contents)
  - [Requirement](#requirement)
  - [Data Preprocessing](#data-preprocessing)
  - [Usage](#usage)
  - [Contributors](#contributors)
  - [References](#references)
  - [Contact](#contact)


## Requirement
```
python >= 3.6
pytorch
embeddings
pkuseg
jieba
```


## Data Preprocessing
For ease of data manipulation, we prefer to do some preliminary processing job before we go further. In this part, we will build vocabulary and generate pretrained weights by using scripts under the `preprocessing` directory:
```shell
  python preprocess/preprocess.py \
                    --data_path /path/to/raw/data \
                    --save_path /path/to/save/data \
                    --key /key/for/version/control
```

Before generating embeddings, we have to ensure that the environmental variable `EMBEDDINGS_ROOT` for `embeddings` is set properly.
```shell
  python preprocess/gen_embed.py \
                    --data_path /path/to/saved/data \
                    --w2v_file /path/to/w2v/file
```


## Usage
The interface switch is under design. Currently, you would have to change the code manually to access different type of models.
For command line user, here are some scripts for invoking a program:
- train.py
- test.py
- etc.

For vscode user, here are some example configurations:
```json
{
    "name": "fsnlu_fscoach_train",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/fsnlu/train.py",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}/fsnlu/",
    "env": {
        "PYTHONPATH": "${workspaceFolder}/fsnlu/",
    },
    "args": [
        "--exp_name", "coach_lstmenc",
        "--exp_id", "ecdt_cbweb_all_7",
        "--bidirection",
        "--freeze_emb",
        "--evl_dm", "cookbook,website",
        "--n_samples", "7",
        "--hidden_dim", "150",
        "--emb_dim", "300",
        "--trs_hidden_dim", "300",
        "--emb_file", "./data/default/token_emb.npy",
        "--slot_emb_file", "./data/default/slot_embs_based_on_each_domain.dict",
        "--tr",
    ]
},
```

```json
{
    "name": "fsnlu_fscoach_test",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/fsnlu/predict.py",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}/fsnlu/",
    "env": {
        "PYTHONPATH": "${workspaceFolder}/fsnlu/",
    },
    "args": [
        "--exp_name", "coach_lstmenc",
        "--exp_id", "ecdt_cbweb_all_7",
        "--bidirection",
        "--freeze_emb",
        "--evl_dm", "cookbook,website",
        "--n_samples", "7",
        "--hidden_dim", "150",
        "--emb_dim", "300",
        "--trs_hidden_dim", "300",
        "--emb_file", "./data/default/token_emb.npy",
        "--slot_emb_file", "./data/default/slot_embs_based_on_each_domain.dict",
        "--model_path", "./fscoach_exp/coach_lstmenc/ecdt_cbweb_all_7/best_model.pth",
        "--model_type", "coach",
        "--tr",
    ]
},
```

## Contributors
* [Changjian Liu](mailto:cjliux@gmail.com) (Harbin Institute of Technology, Shenzhen)
* [Hongru Wang](mailto:hrwang@cse.cuhk.edu.hk) (The Chinese University of Hong Kong)
* et al.

## References
We thanks all authors of the involved models for sharing their code and data. Here are the list of works that we refer to and repositories that we may have heavily borrow from:
- Coach: A Coarse-to-Fine Approach for Cross-domain Slot Filling. ([code](https://github.com/zliucr/coach))
- [Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network](https://atmahou.github.io/attachments/atma's_acl2020_FewShot.pdf) ([code](https://github.com/AtmaHou/FewShotTagging))

## Contact

If you have questions, suggestions and bug reports, please email [cjliux@gmail.com](mailto:cjliux@gmail.com).

