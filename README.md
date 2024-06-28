# SEDA
This repository contains the source code for SEDA.
You can follow the instructions below to reproduce our experiments. Our source code references `https://github.com/shichao-wang/CircEvent.git`

## Dataset
Our experiments are conducted on the New York Times (NYT) portion of the English Gigawords.
You can get access from the [official website](https://catalog.ldc.upenn.edu/LDC2003T05).
The data split we used is provided by Granroth-Wilding[[1]](https://mark.granroth-wilding.co.uk/papers/what_happens_next/)
We annotate the raw documents based on Lee[[2]](https://github.com/doug919/multi_relational_script_learning) with the standford CoreNLP toolkit.
The configuration of CoreNLP is listed in `corenlp.props` file.

## Environment Setup
We conducted our experiments with on a workstation with a tesla V100.
Our programs are tested under PyTorch 1.8.1 + CUDA 10.2.

1. Setup Python environment. We encourage using conda to setup the python virtual environment.
   `conda create -n seda python==3.8 && conda activate seda`
2. Install the CUDA toolkit and Pytorch.
   `pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple``
4. Install the pip packages.
   `pip install -r requirements.txt`
5. Install the circumst_event package
   `pip install -e .`

Now the environment has been set up in the `seda` virtual environment.

## Reproduce Steps
The source codes can be divided into two parts, i.e. data preprocessing and model training.
The entry scripts are placed in `bin` folder. Each step and its corresponding script is listed below.

1. extract text out of gigaword xml file. `1_extract_gigaword_nyt.py`
2. annotate text with CoreNLP. `2_corenlp_annotate.py`
3. extract event chain from annotated document. `3_extract_event_chain.py`
4. convert event chain words to ids. `4_index_event_chain.py`
5. split into train, validation, test set. `5_split_dataset.py`
6. train the circ model. `6_circ_train.py`
7. evaluate the saved model and generate quantitative analysis file. `7_circ_eval.py`
7. draw figures of changing in accuracies based on quantitative analysis file. `8_mask_multiple_items_with_weights.py`


# Reference
[1] Shichao Wang, Xiangrui Cai, Hongbin Wang, and Xiaojie Yuan. 2021. Incorporating circumstances into narrative event prediction. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4840–4849.

[2] Mark Granroth-Wilding and Stephen Clark. 2016. What happens next? Event Prediction Using a Compositional Neural Network Model. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, pages 2727–2733, Phoenix, Arizona, February. AAAI Press.

[3] I-Ta Lee and Dan Goldwasser. 2019. Multi-Relational Script Learning for Discourse Relations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4214–4226, Florence, Italy, July. Association for Computational Linguistics.
