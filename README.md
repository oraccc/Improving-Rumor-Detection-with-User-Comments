## Implementation of RoBERT and ToBERT for Rumour Detection

### 0. Overall Structure

<img src="./images/Overall Structure.png" alt="Overall Structure" style="zoom: 67%;" />

### 1. Requirements
* **Datasets Download**
  * https://drive.google.com/drive/folders/1o430G2HXg9k5cWCOkPwmhOT_7boUii8i?usp=sharing
* **Trained Models Download**
  * https://drive.google.com/drive/folders/1VEtruvbJ9eRMC4BttXgvz9A0h76HgUIP?usp=sharing
* **Python Enviroments**
  * python 3.7
  * numpy 1.21.2
  * pandas 1.3.1
  * pytorch 1.7.0
  * torchtext 0.6.0
  * tensorflow 2.0.0
  * transformers 4.9.2
  * scikit-learn 0.24.2

### 2. Usage Guide

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



### 3. Parameter Settings for Different Models

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

