# Finding-Donors-for-CharityML
實作一個完整的資料專案：
探索性資料分析、建立端到端機器學習模型、建立模型評估機制、根據專案目標選擇最適模型、修改與測試資料性能及調整模型參數。
網頁版 [jupyternotebook](https://nbviewer.jupyter.org/github/JHL01/Finding-Donors-for-CharityML/blob/master/%E5%B0%8B%E6%89%BE%E6%BD%9B%E5%9C%A8%E6%8D%90%E5%8A%A9%E8%80%85%28Finding%20Donors%20for%20CharityML%29.ipynb)
>### 運作環境：
- Python version == 3.6.4
  - sklearn == 0.20.2
  - numpy == 1.15.4
  - pandas == 0.22.0
  - seaborn == 0.9.0
  - scipy== 1.1.0
>### 專案介紹：
CharityML 是一家虛構的慈善組織，位於矽谷的中心地帶，旨在為渴望學習機器學習的人們提供財務支持。在向社區居民發送了近 32,000 封信之後，CharityML 確定他們收到的每一筆捐款都來自每年收入超過 5 萬美元的人。為了擴大他們潛在的捐贈者基礎，CharityML 已決定向加州居民發送信件，但僅向最有可能捐贈給慈善機構的人發信。加州擁有將近 1500 萬名工作者，請幫他們構建算法，以便最好地識別潛在的捐贈者，並降低發送郵件的間接成本。因此，**目標是評估和優化幾個不同的機器學習模型，以確定哪種算法將提供最高的捐贈收益，同時還減少發送的信件總數。**
>### 資料介紹：
修改後的人口普查資料集由大約 32,000 筆資料組成，每筆資料有 13 個特徵。該資料集是由 Ron Kohavi 撰寫的論文 *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid"* 中的資料集的修改版本。其中原始資料集託管在 [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income) 上。
>### 方法概述：
本文利用 [DeepLearning (2016)](http://www.deeplearningbook.org/contents/guidelines.html)書中建議的方法走過機器學習的流程，從快速建立一個簡單的機器學習 Pipeline 開始，藉由評估模型了解優化方向，接著由探索性分析與特徵工程改善特徵，最後優化模型參數獲得更佳的準確率。
>### 主要發現：
- 使用 DeepLearning 書中建議的方法能夠快速測試每次修改的效能變化，從而確定正確的行動過程而不是盲目猜測修改結果，對於一個產品/模型優化而言很有幫助。
- 以本資料集的預測結果來看，Gradient Boosting 具有較高的 F-score(74.5%)，高出 Logistic Regression 4.4%，由學習曲線可知兩者皆不具有overfitting的問題。
- 參數調整後 Gradient Boosting 的 F-score 從 74.5% 提升至 75.1%，提升效果有限，在無overfitting的情況下顯示模型已良好擬合數據，若要增加準確度建議蒐集/生成更多資料。
- 根據特徵重要性分析，較能影響預測結果(與收入相關)的特徵是資本損失、資本收入、教育年數、婚姻狀態及於家庭中的角色，掌握這些少數特徵將有機會快速判斷一個人的收入情況。
