# Tell Me How to Ask Again: Question Data Augmentation with Controllable Rewriting in Continuous Space

Paper: https://www.aclweb.org/anthology/2020.emnlp-main.467.pdf

Conference: EMNLP 2020

Authors: Dayiheng Liu, Yeyun Gong, Jie Fu, Yu Yan, Jiusheng Chen, Jiancheng Lv, Nan Duan, Ming Zhou

## What is this paper about?
Data augmentation (DA)在CV中已經有不錯的成果，但在NLP中，如何改寫文字但仍保有原本語意是一個大的挑戰。
本篇提出的資料增強方法為 Controllable Rewriting based Question Data Augmentation (CRQDA), 增強的資料集可以提升machine reading comprehension (MRC)等任務的performance。著重在question 的重寫，以生成與上下文相關，高品質且多樣化的question data samples。
簡單來說要解決的問題就是：產生answerable and unanswerable questions for data argumentation.

## Tech Detail
![](img1.png)

整個CRQDA有兩個components，Trained MRC model和Transformer Autoencoder。
Trained MRC model就是一般的pretrained model，直接理解為BERT等語言模型用在QA任務就好。上圖左就是一般QA任務的模型架構，而這篇的重點是question embedding($\bm{E}^q$)，train好這個模型後，把$\bm{E}^q$放入 Transformer encoder的 embedding layer，然後訓練Autoencoder時固定權重。這麼做是要讓兩個模型有相同的continuous embedding space.

到目前為止，結論一下這兩個模型目前的功用，Trained MRC model是為了產生跟QA相關的embedding，而Transformer autoencoder是用來生成問題用的。所以，我們只要調整了question embedding($\bm{E}^{q^{\prime}}$)，就可以透過Transformer autoencoder來生成新的question data($\hat{q}^\prime$)

在生成新的question($\hat{q}^\prime$)前我們要釐清一些事，如果我們給一個answerable question，那麼我們要生成一個跟原文有相關的unanswerable question，有幾個目標必須確定：
1. 修改後的question embedding應使Trained MRC model從有答案變無答案
1.  revised question($\hat{q}^\prime$)應該跟原始問題$q$很相近，這樣才能提升模型的robustness




## What contributions does it make?

## What are the main strengths?

## What are the main weaknesses?

## Scores