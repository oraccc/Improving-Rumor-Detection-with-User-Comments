## Improving Rumour Detection with User Comments



### 1. Project Introduction

- In this project, we propose a new **BERT-based** rumour detection method considering **both the original post and the associated comments**. 
- The method starts with concatenation of the original post and the associated comments to form a single long text, which is then segmented into shorter sequential chunks more suitable for BERT-based vectorization. Features extracted by applying BERT to all the trunks are fed into an LSTM- or transformer-based classifier for the binary ('rumour' or 'non-rumour') classification task. 
- The experimental results on **PHEME** and **Ma-Weibo**, two public rumour detection datasets representing the two most spoken languages -- English and Chinese --  and two of the largest Web 2.0 platforms -- Twitter and Sina Weibo, showed that our method outperformed other state-of-the-art methods, mostly with a significant margin.

### 2. Implementation Requirements & Preparations

* **Data Preparations**

  Please download the datasets and trained models from the google drive links below and place these file folders in the corresponding working directories. 

  * Datasets Download Link 
    * https://drive.google.com/drive/folders/1o430G2HXg9k5cWCOkPwmhOT_7boUii8i?usp=sharing

  * Trained Models Download Link (Not Necessary)
    * https://drive.google.com/drive/folders/1VEtruvbJ9eRMC4BttXgvz9A0h76HgUIP?usp=sharing


* **Recommended Environments to Run Codes (Win/Linux)**
  * python 3.7
  * numpy 1.18.5
  * pytorch 1.7.0
  * torchtext 0.6.0
  * tensorflow 2.0.0
  * transformers 4.9.2

### 3. Usage Guide

* **Data_Preprocess.ipynb**
  * This notebook will preprocess the original dataset and generate a **'.csv'** file in **'./data'** folder. Our model will only accept the '.csv' file as input. 
  * I have already generated a **'raw_data.csv'** file so there is no need to collect the dataset and run this notebook again.  Just download the csv file ,head to 'Rumour_BERT.ipynb' and run the model.

* **Rumour_BERT.ipynb**

  - [x] Step 1: Process the 'raw_data.csv' and generate the original train and test data.

  - [x] Step 2: Define the BERT model for features extraction and classification.
  - [x] Step 3 & 4: Fine-tune, train,  save and evaluate the BERT model.
  - [x] Step 5: Get the text embeddings and prepare datas for RoBERT and ToBERT.
  - [x] Step 6: Implement, train, save and evaluate the **RoBERT**.
  - [x] Step 7: Implement, train, save and evaluate the **ToBERT**.

  **Some Explanations:**

  - The order of Step 6 and Step 7 is not mandatory.
  - The accuracy result in Step 4 is only for the BERT model, it is **NOT** the final result.
  - **The final result is in Step 6 and 7's output**, measured with four indicators.
  - You can use my **trained model** so you don't have to train and fine-tune the BERT model again, you will only need to evaluate the RoBERT and ToBERT part, which will save you a lot of time :)
    - Please put the trained model in '**./trained_models**' folder.
    - **Skip Step 3 & Step 4**.
    - Select the '**model_path**' in Step 1 and other parameters to evaluate the models. For more detailed description, see below.



### 4. Parameter Settings for Different Models

* **text_comments** (Default)

  * Set the 'model_path' to **'text_comments'** (Default).

  * **Don't** uncomment lines in Cell *"\## Different Number of Comments ##"*.

  * Use the following lines in Cell *"\## Data Selection ##"*.

    ```python
    raw_data = raw_data[['text_comments','label']]
    raw_data = raw_data.rename(columns = {'text_comments':'text'})
    ```

* **text_only, comments_only**

  * Set the 'model_path' to **'text_comments'** or **'comments_only'**.

  * **Don't** uncomment lines in Cell *"\## Different Number of Comments ##"*.

  * Use the following lines in Cell *"\## Data Selection ##"*. Don't forget to comment the rest lines in the cell.

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

      

* **comments_group1, comments_group2, comments_group3**

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

  * Use the following lines in Cell *"\## Data Selection ##"*.

    ```python
    raw_data = raw_data[['text_comments','label']]
    raw_data = raw_data.rename(columns = {'text_comments':'text'})
    ```

* **natural_split**
  * Set the 'model_path' to **'natural_split'**.
  * **Don't** uncomment lines in Cell *"\## Different Number of Comments ##"*.

  * Use the following lines in Cell *"\## Data Selection ##"*.

    ```python
    raw_data = raw_data[['text_comments','label']]
    raw_data = raw_data.rename(columns = {'text_comments':'text'})
    ```
  
  * In Step 1.2, Use **'get_natural_split'** function rather than 'get_split', please don't forget to comment the other code line.

### 5. Results
* PHEME Dataset

  |        | Accuracy    | Precision   | Recall      | F1-Score    |
  | ------ | ----------- | ----------- | ----------- | ----------- |
  | RoBERT | 0.96301     | 0.94521     | **0.94694** | 0.94607     |
  | ToBERT | **0.96287** | **0.95118** | 0.94283     | **0.94670** |

* Ma-Weibo

  |        | Accuracy    | Precision   | Recall      | F1-Score    |
  | ------ | ----------- | ----------- | ----------- | ----------- |
  | RoBERT | 0.98075     | **0.97252** | 0.98785     | 0.98011     |
  | ToBERT | **0.98128** | 0.97185     | **0.99022** | **0.98093** |
