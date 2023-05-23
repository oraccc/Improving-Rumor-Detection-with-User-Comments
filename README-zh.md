## 使用用户的评论提升谣言检测的性能

**[English](https://github.com/oraccc/Improving-Rumor-Detection-with-User-Comments/blob/main/README.md)** | **中文**



> 该repo是 [TrustCom2022](http://www.ieee-hust-ncc.org/2022/TrustCom/) 论文 " 'Comments Matter and The More The Better': Improving Rumor Detecion with User Comments" 的源码.
>
> [文章链接](https://ieeexplore.ieee.org/document/10063596)

### 1. 项目介绍

- 在这个项目中，我们提出了一种**基于BERT**的谣言检测方法，考虑到**原始帖子和相关评论**的内容。

- 该方法首先将原始帖子和相关评论拼接起来形成一个长文本，然后将其分割成更适合BERT特征化的短序列块。这些短序列块的特征会在拼接后被送到基于LSTM或Transformer的分类器中，用于进行二元（即`"谣言"`或`"非谣言"`）分类任务。

- 在代表两种最常用语言（英文和中文）和两个最大的Web 2.0平台（Twitter和新浪微博）的公共谣言检测数据集 **PHEME** 和 **Ma-Weibo** 上进行的实验证明，我们的方法在性能表现优于其他SOTA方法，而且优势明显。

- 模型整体架构

<div align=center>
  <img src="https://raw.githubusercontent.com/oraccc/Improving-Rumor-Detection-with-User-Comments/main/images/Overall-Structure.png" width="750"/>
</div>


---

### 2. Implementation Requirements & Preparations

* **Data Preparations**

  Please download the datasets and trained models from the google drive links below and place these file folders in the corresponding working directories. 

  * [Datasets Download Link](https://drive.google.com/drive/folders/1o430G2HXg9k5cWCOkPwmhOT_7boUii8i?usp=sharing)

  * [Trained Models Download Link](https://drive.google.com/drive/folders/1VEtruvbJ9eRMC4BttXgvz9A0h76HgUIP?usp=sharing) 

* **Folder Structure**


  ```shell
  ├─images
  │  └─***.png
  ├─Ma-Weibo
  │  ├─Data_preprocess.ipynb
  │  ├─Rumor_BERT.ipynb
  │  ├─utils.py
  │  ├─data
  │  │  └─raw_data.csv
  │  └─trained_models
  │     ├─classification_models_text_comments
  │     └─...
  └─PHEME-RNR
      ├─Data_preprocess.ipynb
      ├─Rumor_BERT.ipynb
      ├─utils.py
      ├─data
      │  └─raw_data.csv
      └─trained_models
         ├─classification_models_text_comments
         └─...
  ```

  

* **Recommended Environments to Run Codes (Win/Linux)**
  
  * python 3.7 or higher
  * numpy 1.18.5
  * pytorch 1.7.0 (with CUDA Version 11.5)
  * torchtext 0.6.0
  * tensorflow 2.0.0
  * transformers 4.9.2

---

### 3. Implemetation Guide

There are two jupyter notebooks in each folder. The `"Data_Preprocess.ipynb"` notebook converted the original dataset (i.e., Ma-Weibo and PHEME) to *"raw_data.csv"* files, while the `"Rumor_BERT.ipynb"` performed the rumor detection and binary (i.e., `'rumor'` or `'non-rumor'`) classification task. The specific introduction is as follows.

* **Data_Preprocess.ipynb**

  * This notebook preprocessed the original dataset and generated a *".csv"* file in *"./data"'* folder. 
  * A **'raw_data.csv'** file has already been generated in each folder in [datasets download link](https://drive.google.com/drive/folders/1o430G2HXg9k5cWCOkPwmhOT_7boUii8i?usp=sharing), so you may alternatively skip this notebook after download the dataset and proceed to *"Rumor_BERT.ipynb"*.

  ---

* **Rumor_BERT.ipynb**

  There are 7 steps in this jupyter notebook, the final result is in Step 6 (RoBERT) and Step 7 (ToBERT) output, measured with four indicators (i.e., *accuracy*, *precision*, *recall* and *f1-score*).

  * Step 1: Process the *"raw_data.csv"* and generate the original train and test data.
  * Step 2: Define and construct the BERT model for text classification.
  * Step 3 & 4: Fine-tune, train and evaluate the BERT model.
  * Step 5: Get the text embeddings and prepare datas for the final classification model.
  * Step 6: Construct, train and evaluate the **RoBERT** (Recurrence over BERT).
  * Step 7: Construct, train and evaluate the **ToBERT** (Transformer over BERT).

  *Some Extra Explanations:*

  - *The accuracy result in Step 4 is only for the BERT model, it is **NOT** the final result.*
  - *You could alternatively use my own "pre-trained"  BERT models so you don't have to train and fine-tune the BERT model again.*
    - *Place my trained model (links above) in "**./trained_models**" folder.*
    - ***Skip Step 3 & Step 4**.*

---

### 4. Settings for Different Model Path (Additional Experiments)

We conducted additional experiments on different settings of our proposed method to study diferent aspects of the role comments play in the rumor detection task. These additional experiments led to some very interesting findings, including further evidence that including the associated comments is beneficial, the surprising result that fixed-length segmentation with an overlap is better than natural segmentation, and the observation that the more comments the better the detector's performance. 

To reproduce these experiment results, please change the **'model_name'** in "Rumor_BERT.ipynb" to coressponding values. Optional values and the settings are listed below.

* **text_comments** (Default), **text_only**, **comments_only**

  * Set the 'model_path' to **'text_comments'** (Default), **'text_comments'** or **'comments_only'**.

  * Use the following lines in notebook cell *"\## Data Selection ##"*.

    * text_comments
    
      ```python
      raw_data = raw_data[['text_comments','label']]
      raw_data = raw_data.rename(columns = {'text_comments':'text'})
      ```
    
    * text_only
    
      ```python
      raw_data = raw_data[['text_only','label']]
      raw_data = raw_data.rename(columns = {'text_only':'text'})
      ```
    
    
    * comments_only
    
      ```python
      raw_data = raw_data[['comments_only','label']]
      raw_data = raw_data.rename(columns = {'comments_only':'text'})
      ```
    
      

* **comments_group1**, **comments_group2**, **comments_group3**

  * Set the 'model_path' to **'comments_group1'**, **'comments_group2'** or **'comments_group3'**.

  * Uncomment the corresponding lines in Cell *"\## Different Number of Comments ##"*.

    * comments_group1

      ```python
      # For 'Ma-Weibo' Dataset
      raw_data = raw_data[raw_data['count'] <= 70]
      
      # For 'PHEME-RNR' Dataset
      raw_data = raw_data[raw_data['count'] <= 7]
      ```

    * comments_group2

      ```python
      # For 'Ma-Weibo' Dataset
      raw_data = raw_data[raw_data['count'] > 70]
      raw_data = raw_data[raw_data['count'] <= 224]
      
      # For 'PHEME-RNR' Dataset
      raw_data = raw_data[raw_data['count'] > 7]
      raw_data = raw_data[raw_data['count'] <= 18]
      ```

    * comments_group3

      ```python
      # For 'Ma-Weibo' Dataset
      raw_data = raw_data[raw_data['count'] > 224]
      
      # For 'PHEME-RNR' Dataset
      raw_data = raw_data[raw_data['count'] > 18]
      ```

* **natural_split**, **fixed_split**
  
  * Set the 'model_path' to **'natural_split'** or **"fixed_split"**.
  * In Step 1.2, Use **'get_natural_split'** or **'get_fixed_split'** function rather than 'get_split'.



