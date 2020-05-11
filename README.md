# DL-Sentiment-Analysis
## Introduction
Through this project, we study the Bidirectional Encoder Representation from Transformers (BERT) and apply it specifically for the task of sentiment classification. We compare the results of the model along with 3 other state of the art models namely,
* FastText
* Char-CNN
* Text-CNN 

## To setup the environment
`cd scripts && bash setup.sh`

## To train the models and get the results.
`python main.py`

## References:
* A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, "Bag of tricks for efficient text classification", arXiv preprint arXiv:1607.01759, 2016.
* X. Zhang, J. Zhao, Y. LeCun, "Character-level convolutional networks for text classification", In Advances in Neural Information Processing Systems, 2015 .
* Y. Kim, "Convolutional neural networks for sentence classification" arXiv preprint arXiv:1408.5882, 2014.
* J. Devlin, M. W. Chang, K. Lee, K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805, 2018.
* A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, “Attention is all you need”, In Advances in neural information processing systems, 2017.

## Acknowledgements.
This work was done as a part of CS 8750 Artificial Intelligence II final project, under the supervision of Dr. Yi Shang.