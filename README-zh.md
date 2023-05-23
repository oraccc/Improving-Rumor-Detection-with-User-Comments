## 使用用户的评论提升谣言检测的性能

**[English](https://github.com/oraccc/Improving-Rumor-Detection-with-User-Comments/blob/main/README.md)** | **中文**

![PyTroch](https://img.shields.io/badge/PyTorch-1.7.0-brightgreen) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0.0-green) ![Transformers](https://img.shields.io/badge/Transformers-4.9.2-yellowgreen)



该repo是 [TrustCom2022](http://www.ieee-hust-ncc.org/2022/TrustCom/) 论文 " 'Comments Matter and The More The Better': Improving Rumor Detecion with User Comments" 的源码.

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

### 2. 代码运行准备

* **数据集准备**

  请从下面的Google Drive链接下载数据集和训练好的模型，并将这些文件夹放置在相应的工作目录中。

  * [数据集下载链接](https://drive.google.com/drive/folders/1o430G2HXg9k5cWCOkPwmhOT_7boUii8i?usp=sharing)

  * [训练好的模型下载链接](https://drive.google.com/drive/folders/1VEtruvbJ9eRMC4BttXgvz9A0h76HgUIP?usp=sharing) 

* **文件夹结构**


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

  

* **推荐的代码运行环境 (Win/Linux)**
  * python 3.7
  * numpy 1.18.5
  * pytorch 1.7.0 (with CUDA Version 11.5)
  * torchtext 0.6.0
  * tensorflow 2.0.0
  * transformers 4.9.2

---

### 3. 代码运行指导

在每一个文件夹下均有两个 Jupyter Notebook 文件。 `"Data_Preprocess.ipynb"` 文件将原始的谣言数据集 (即Ma-Weibo 与 PHEME) 转换为 `"raw_data.csv"` 文件,  `"Rumor_BERT.ipynb"` 文件实现了主要的谣言检测与二分类任务（即`"谣言"`或`"非谣言"`），具体的运行指导如下：

* **Data_Preprocess.ipynb**

  * 该 notebook 处理原始的数据集数据并生成一个 *".csv"* 文件在 *"./data"* 文件夹下. 
  * 在[数据集下载链接](https://drive.google.com/drive/folders/1o430G2HXg9k5cWCOkPwmhOT_7boUii8i?usp=sharing)中的每个文件夹中已经生成了一个名为 **'raw_data.csv'** 的文件，因此您可以选择跳过此 notebook，在下载数据集后继续进行 *"Rumor_BERT.ipynb"*。

  ---

* **Rumor_BERT.ipynb**

  此 notebook 共有7个步骤，最终结果在第6步（RoBERT）和第7步（ToBERT）的输出中，使用四个指标进行评估（即准确率、精确率、召回率和F1值）。

  * 第1步：处理 "raw_data.csv" 文件并生成原始的训练和测试数据。
  * 第2步：定义并构建用于文本分类的BERT模型。
  * 第3步和第4步：对BERT模型进行微调、训练和评估。
  * 第5步：获取文本嵌入并准备最终分类模型的数据。
  * 第6步：构建、训练和评估RoBERT。
  * 第7步：构建、训练和评估ToBERT。

  *一些额外的解释：*

  - *第4步的准确率结果仅针对BERT模型，它**不是**最终结果。*
  - *你可以选择使用我自己训练好的BERT模型，这样就不必再次训练和微调BERT模型。*
    - *将我的训练好的模型（上面的链接）放置在“./trained_models”文件夹中。*
    - ***跳过第3步和第4步***。

---

### 4. 对不同 Model Path 的设置(额外的实验)

我们对我们提出的方法的不同设置进行了额外的实验，以研究评论在谣言检测任务中的不同作用。这些额外的实验得出了一些非常有趣的发现，包括进一步证据表明包括相关评论是有益的，固定长度的分割与重叠比自然分割更好的意外结果，以及观察到评论越多，检测器的性能越好。

要重现这些实验结果，请在"Rumor_BERT.ipynb"中更改**'model_name'**为相应的值。可选的值和设置如下所示。

* **text_comments** (默认值), **text_only**, **comments_only**

  * 将'model_path'设置为**'text_comments'**（默认值），**'text_comments'**或**'comments_only'**。

  * 在笔记本单元格*"## Data Selection ##"*中使用以下代码行。

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

  * 将'model_path'设置为**'comments_group1'**、**'comments_group2'**或**'comments_group3'**。

  * 在单元格*"## Different Number of Comments ##"*中取消注释相应的代码行。

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
  
  * 将'model_path'设置为 **'natural_split'** 或 **'fixed_split'**。
  * 在第1.2步中，使用 **'get_natural_split'** 或 **'get_fixed_split'** 函数，而不是使用 'get_split' 函数。



