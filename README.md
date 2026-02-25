# LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers
[![arXiv](https://img.shields.io/badge/arXiv-2503.14434-b31b1b.svg)](https://arxiv.org/abs/2503.14434)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-LLMFE-yellow)](https://huggingface.co/papers/2503.14434)

Official implementation of  [LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers.](https://arxiv.org/abs/2503.14434)

![](llmfe.jpg)

## 📄 Overview
LLM-FE is a novel framework that leverages Large Language Models (LLMs) as evolutionary optimizers to automate feature engineering for tabular datasets.  LLM-FE iteratively generates and refines features using structured prompts, selecting high-impact transformations based on model performance. This approach enables the discovery of interpretable and high-quality features, enhancing the performance of various machine learning models across diverse classification and regression tasks.

## ⚙️ Installation
To run the code, create a conda environment and install the dependencies using `requirements.txt`:

```
conda create -n llmfe python=3.11.7
conda activate llmfe
pip install -r requirements.txt
```

## 🔧 Usage

In `run_llmfe.sh` file, set the OPENAI API key under 
```
export API KEY = <ENTER YOUR API KEY>
```

To run the LLM-FE pipeline on a sample dataset:
```
bash run_llmfe.sh
```

### ✅ Evaluation

- The **generated features** for each dataset are stored in the logs under the `samples/` folder.
- To **run the evaluation**, open the `evaluation.ipynb` notebook and set the `pb_name` variable to the dataset name you want to evaluate (i.e., replace the existing dataset name in `pb_name` with your target dataset), then run the notebook cells.

## 📝 Citation
```
@article{abhyankar2025llm,
  title={LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers},
  author={Abhyankar, Nikhil and Shojaee, Parshin and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2503.14434},
  year={2025}
}
```

## 📄 License

This repository is licensed under MIT licence.

This work is built on top of other open source projects like [FunSearch](https://github.com/google-deepmind/funsearch) and [LLM-SR](https://github.com/deep-symbolic-mathematics/llm-sr). We thank the original contributors of these works for open-sourcing their valuable source codes.


## 📬 Contact Us
For any questions or issues, you are welcome to open an issue in this repo, or contact us at  [nikhilsa@vt.edu](nikhilsa@vt.edu) and [parshinshojaee@vt.edu](parshinshojaee@vt.edu).
