# A Survey of Transformers
本篇探討在vanilla transformer提出之後各種transformer的改進

## Vanilla Transformer


Vanilla Transformer是目前NLP領域中幾乎佔主導的深度學習基本網路架構(相對於LSTM, CNN而言)，各種pretrained model幾乎都是由Transformer所組成，即便Vanilla Transformer取得了許多成功，但為人詬病的點仍然很明顯，就是他的計算與空間複雜度，所以許多針對改進Vanilla Transformer的論文也不斷出現，而這些Transformers也同樣成為新的pretrained model的核心架構。

先來聊聊Vanilla Transformer的複雜度吧

首先self-attention的計算複雜度為 $O(hdn^2)$，$h$為attention的head數，$d$為query和key的維度，$n$為句子長度， dot-product 公式為
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V $$

由以上公式再乘上head數就可以推出計算複雜度。

同樣的空間複雜度也受到句子長度影響，為$O(hdn+hn^2)$，前者為存query和key所需的空間，後者為每個head產生的attention matrix。

我們以BERT-Base為例計算一下複雜度，BERT-Base句子長度為512，hidden size 768，12 heads，所以每個head維度為64 (768 / 12)。在這設定下，393216 floats(12 heads * 64 head size * 512 sequence length) = 393216 * 4 bytes = 1572864 bytes = 1572864/1024/1024 MB = 1.5 MB，1.5MB為query和key所需的空間。每個head產生的attention matrix所需空間為3145728 floats (12 * 512 * 512)～12MB，幾乎為前者的10倍大。BERT-Base為12層，所以每個example所需的memory接近(12+1.5)*12 = 162MB，當句子長度為1024時，$(12\times(1024^2)\times4/1024/1024+12\times64\times1024\times4/1024/1024)\times12 = 612$ MB，需要那麼多memory。

這也意味著在訓練時，我們只能用更小的batch size以及較差的平行運算能力，這也導致模型對長文本的推理能力較差。

---

## Sparse Transformers

Generating Long Sequences with Sparse Transformers

Paper: https://arxiv.org/pdf/1904.10509.pdf

Authors: Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever 

這篇是OpenAI在2019年提出的新的transformer架構，其目的就是為了解決Vanilla Transformer的空間複雜度問題，將原本的$O(n^2)$降到$O(n\sqrt{n})$。

![](img1.png)

直接從圖簡單理解，b,c兩張圖是這篇paper提出的兩個方法，而想法為，不同於傳統的方法一次要跟所有token做attention，他們採用分組的方式，但組跟組之間也不是完全沒有關聯，淺藍色負責的就是整個sequence的attention。所以同一組之間的attention是強烈的，而長距離的attention可以透過加深神經網路的方式，讓相關的訊息互通，如此一來既能夠達到全局的attention也可以減少memory的使用。這樣的方法允許模型處理規模非常大的上下文。

有一個疑問是，既然全局的attention需要用層層疊加的，那麼效率上是否會有影響呢？但或許Vanilla Transformer詬病的就是空間複雜度的問題，長文本的運算cost非常高，這篇的想法就是將降低空間複雜度視為當務之急吧！

---

## Reformer: The Efficient Transformer

Paper: https://arxiv.org/abs/2001.04451

Conference: ICLR 2020

Authors: Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya

上面提到說Vanilla Transformer能力雖強，但缺點也很明顯，就是在超長的文本中，需要花費太多的memory去存模型的參數，包含backpropagation中所需的參數。另外transformer的關鍵就是softmax，如果在超長文本中，長度為N，那麼模型每一步就要理解NxN對單詞之間的關係，非常不make sense。
而Reformer就是來解決上述提到的softmax和memory的問題，將 dot-product attention的計算複雜度降低，以及減少memory的使用，分別是兩個技術locality-sensitive hashing(LSH)以及Reversible Residual Network。實驗結果也證實了Reformer跟Transformer有同等的性能(這裡指準確率的指標)，且達到更好的記憶體使用效率以及對應更長文本能夠有效且快速地學習。

###  Locality-Sensitive Hashing Attention

在論文中先大概介紹了一下transformer中的Dot-product attention，並指出transformer中存在的弱點，其中一個討論很有趣，就是**Where do Q, K, V come from?**。我在理解transformer上對於為何要用QKV是這樣理解的，同一個tensor只是分別經過三個不同的線性轉換，轉成我們人類所定義的QKV，以符合Dot-product attention的公式。而在 LSH attention中，QK用同一個linear(即Q=K)，V用另一個，理解為QK的定義上更為相似，既然要省記憶體，那麼相似的事物共享同個神經網路也不為過吧。

#### Hashing Attention
在attention的公式中，$QK^T$的shape為[batch size, length, length]，但我們真正在意的是$softmax(QK^T)$，是softmax的結果，所以以更宏觀的角度來看，只要能達成softmax上想要表達的意義，那麼就不一定真的要去計算NxN對單詞的關聯。進一步的，我們把key分群(即把整個tensor拆成多段)，一個query就關注某一群跟當前query有關的K就好，而不是全部的keys。


#### Locality sensitive hashing
所以從上面的想法推廣出來，一個向量$x$透過hash $h(x)$將相似的向量放到同個hash-bucket

![](img2.png)

從上圖可以更快速理解，LSH透過計算hash func來實現這一點，該hash func將類似的向量配對在一起，而不是搜索所有可能的向量對。
不同的顏色表示不同的hash-bucket，相似的單詞有相同的顏色。當hash value被分配時，sequence會被重新排列，同個hash value會被分在同一塊，每一塊長度一樣，然後排序，可以平行計算。然後將attention放在這些更短的chunks(以及它們的相鄰塊有同個hash value)中，從而大大減少了計算複雜度。

