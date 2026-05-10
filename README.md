<h1 align="center">Total Reading Time Prediction from Eye-Tracking Data</h1>

<p align="center">
  Șincari Sebastian George<br>
  <a href="mailto:sebastian-george.sincari@s.unibuc.ro">sebastian-george.sincari@s.unibuc.ro</a>
</p>

---

## Abstract

This study investigates the prediction of word-level Total Reading Time from eye-tracking data collected on Romanian texts. The dataset contains only the word, its position index, the text ID, and the total reading time, so a preprocessing pipeline is built to extract lexical, morpho-syntactic, and contextual features. Four models are evaluated: a LightGBM regressor, a Hybrid LightGBM, a FeedForward Neural Network, and a Stacked Ensemble of gradient boosting models. The results show that the Stacked Ensemble achieves the best performance, with a custom score of 38.86 on the test set, confirming that combining multiple models produces more robust predictions than any single approach.

---

## 1 Introduction

Reading time prediction is the problem of estimating how long a person looks at a word while reading a sentence. This is not a simple task, beacause reading speed is influenced by many factors at the same time, such as word frequency, syntactic complexity, and the predictability of a word given its context. Understanding these factors is important for natural language processing, because it connects the way models process text with the way humans actually read it.

This study was developed as a semester project, but the subject is treated as a personal research problem. The reason is that the combination of cognitive signals and machine learning is, in my opinion, one of the most interesting directions in modern NLP. Eye-tracking data gives a direct measurement of human reading behavior, and the challenge of predicting it from linguistic features alone is both technically difficult and scientifically meaningful.

The specific task is to predict word-level Total Reading Time for Romanian texts, using a dataset collected with an Eye Tracker. This is a particularly difficult setting, beacause Romanian is a morphologically rich language with flexible word order, and public cognitive corpora for Romanian did not exist until very recently (Hodivoianu et al., 2025).

The preprocessing pipeline, the models, and the evaluation are described in detail in Section 3. The results and a discussion of their implications are presented in Section ??.

---

## 2 Related Work

Eye-tracking data has become a useful tool in natural language processing, because it shows how humans read and process text. Most early work on predicting reading times used simple regression models with handcrafted features like word frequency and word length. However, this type of research for the Romanian language was very limited, beacause no public eye-tracking dataset existed.

This changed with (Hodivoianu et al., 2025), who released the first Romanian eye-tracking dataset for reading and established the current state-of-the-art for predicting word-level Total Reading Time. Their study showed that combining linguistic features with contextual surprisal inside ensemble models produces strong results, even when compared to more complex neural approaches.

---

## 3 Methodology

### 3.1 Dataset

The dataset used in this study was collected using an Eye Tracker. The subjects of this experiment had to read ten, plus two for accomodation with the setup. As an approval for data that was collected from them, they had to respond at six questions after each text. This ensures that the data are valid and reflect genuine, attentive reading, confirming that participants were engaged and did not merely simulate reading.

However, the dataset used in this study does not include raw eye-tracking metrics; it contains only the word, its position index on the page, the text ID, and the total reading time. In this case, the preprocessing step in which to extract more features is absolutly necessary.

### 3.2 Preprocessing

The features extracted from the dataset used in this study can be categorized in following classes: lexical, morpho-syntactic and contextual features.

#### 3.2.1 Lexical features

Lexical feature extraction process is fairly direct, extracting word length (including punctuation), squared word length ment to model non-linear effects, number of diacritics (specific of Romanian words), binary indicator if it is capitalized, binary indicator for sentence-initial position, binary indicator for punctuation at the end of the word, binary flag for presence of non-alphabetic characters. In addition, other extracted lexicat feature, based on Zipf scale, word frequency score in Romanian; using spaCy (Honnibal and Montani, 2017) a binary indicator for Romanian stopword, number of syllables count using a heuristic vowel-group method (count transitions from non-vowel to vowel in "aeiouăâî") and a vowel-consonant ratio ((# vowels) / (word length)).

#### 3.2.2 Morpho-Syntactic Features

Morpho-syntactic features are extracted using spaCy (Honnibal and Montani, 2017) after reconstructing the text. Each word is assigned a part-of-speech tag (spaCy) and a dependency relation (spaCy), both encoded as numeric values. The dependency depth is also computed (spaCy) as the number of ancestors in the syntax tree, used as a simple measure of syntactic complexity.

A binary indicator for content words is included, based on the tags `NOUN`, `VERB`, `ADJ`, `ADV`, `PROPN` (spaCy). From named entity recognition, a binary feature shows if a word is part of a named entity, together with its type (spaCy).

Morphological features include the number of attributes (spaCy), grammatical case (encoded as `Nom`, `Acc`, `Dat`, `Gen`, `Voc`, extracted with spaCy), and verb form (encoded as `Fin`, `Inf`, `Part`, `Ger`, `Sup`, also from spaCy). A binary feature also indicates if the word is inflected (comparison between token and lemma, spaCy).

#### 3.2.3 Contextual Features

Contextual features are computed using a pretrained Romanian language model, RoGPT2 (Card, 2020) (via `transformers`). The text is tokenized (`transformers` tokenizer) and processed in chunks. For each word, a surprisal score is calculated as the negative log-probability (base 2) of its tokens given the context (RoGPT2).

This surprisal value reflects how predictable a word is in context and is linked to reading difficulty (Hale, 2001; Levy, 2008).

Parafoveal features are also included by shifting values within each text and page (`pandas`). These include previous and next word surprisal (from RoGPT2), length (basic string processing), and frequency (from `wordfreq`), capturing local context effects.

At a higher level, a relative position feature is computed as the word index divided by total words in the page (`pandas`). A type-token ratio is calculated per text using grouping operations (`pandas`) to measure lexical diversity. Finally, an interaction between surprisal and frequency is added: `surprisal * log2(1 + frequency)` (NumPy).

### 3.3 Method

The prediction of reading time is formulated as a regression task, where the target variable is the total reading time per word, as recorded by the Eye Tracker. Four models are evaluated in this study: a LightGBM regressor, a Hybrid LightGBM, a Feed-Forward Neural Network, and a Stacked Ensemble.

#### 3.3.1 LightGBM

LightGBM is a gradient boosting method that builds decision trees sequentially, where each new tree corrects the errors of the previous one. This makes it well suited for tabular data with mixed feature types, such as the combination of continuous linguistic scores and categorical morpho-syntactic tags used in this study. (Ke et al., 2017)

The feature matrix contains lexical and linguistic numeric features (for example `word_len`, `word_len_sq`, `n_diacritics`, `zipf_frequency`, `n_syllables`, `vowel_consonant_ratio`, `is_stopword`, `is_NE`, `is_content_word`, `contextual_surprisal`, `morph_count`, `is_inflected`). To extend the contextual window available to the model, lag and lead features are computed for `word_len`, `contextual_surprisal`, and `zipf_frequency` at offsets t−1, t−2, t+1, and t+2 within each `participant_id`/`text` group, capturing the effect of neighboring words on the reading time of the current one.

Validation uses a participant-level split, meaning the model is evaluated on unseen readers, not unseen words. The model is trained with hyperparameters `num_leaves`, `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`, tuned manually. Early stopping on the validation set is applied to find the optimal number of trees and prevent overfitting.

#### 3.3.2 Hybrid LightGBM

The Hybrid LightGBM pipeline extends the standard approach by splitting the prediction into two stages. First, a binary classifier predicts whether the reading time is greater than zero, using `answer > 0` as the label. Second, a regression model is trained only on samples with positive reading time values. At inference time, the final output is the product of the classifier prediction, after thresholding at 0.5, and the regressor output. In other words, if the classifier decides a word was not fixated, the predicted reading time is zero. The same feature preparation, lag and lead features, and participant-level validation split are used in both stages.

#### 3.3.3 Feed-Forward Neural Network

The Feed-Forward Neural Network is a tabular regression model built as a multilayer perceptron (a stack of linear transformations followed by non-linear activations, allowing the model to capture complex interactions between features). It uses only numeric features, standardized using the mean and standard deviation from the training set.

The architecture has three hidden layers with sizes 512, 256, and 128 neurons. Each hidden layer is followed by batch normalization (which stabilizes training by normalizing the activations), a ReLU activation, and dropout regularization to prevent co-adaptation of neurons. The output layer is a single linear neuron, appropriate for regression.

The target is transformed with `log1p` during training and converted back with `expm1` at inference time, which reduces the effect of large outliers in reading time. The model is trained with the Adam optimizer and Huber loss (a loss function that is less sensitive to outliers than mean squared error). A learning-rate scheduler reduces the learning rate when validation performance stops improving, and training stops early when no improvement is observed for several consecutive epochs.

#### 3.3.4 Stacked Ensemble

The Stacked Ensemble combines three gradient boosting regressors: CatBoost (Prokhorenkova et al., 2018), LightGBM (Ke et al., 2017), and XGBoost (Chen and Guestrin, 2016). Each base model is trained using GroupKFold cross-validation with `participant_id` as the grouping variable, ensuring that each validation fold contains only unseen participants. The same feature preparation, including lag and lead features and categorical encoding, is applied to all three models.

For each fold, the three base models produce out-of-fold predictions on the validation participants. These predictions are stacked into a new feature matrix, which is used to train a Ridge meta-model (a linear regression with L2 regularization that learns the optimal weight for each base model's contribution). The final prediction is the output of this meta-model. For the test set, predictions from each base model are averaged across all folds before being passed to the Ridge regressor.

#### 3.3.5 Evaluation

Model performance is evaluated using four metrics. RMSE (Root Mean Squared Error) measures the average magnitude of prediction errors in the same unit as reading time. R² measures the proportion of variance in reading time explained by the model. Pearson correlation measures the linear agreement between predicted and actual values. Finally, a composite score is computed as:

$$\text{Score} = 100 \cdot \frac{R^2 + |\rho|}{2}$$

where ρ is the Pearson correlation. This score balances explained variance with rank-order agreement and serves as the primary evaluation criterion.

#### 3.3.6 Results

The table below summarizes validation metrics for each model and the custom score on the test set. Validation metrics are computed on the held-out participant validation set.

**Table 1: Validation metrics and final test score**

| Metric | LightGBM | Hybrid LightGBM | Neural Network | Stacked Ensemble |
|:---|:---:|:---:|:---:|:---:|
| RMSE (val) | 285.9981 | 289.6788 | 233.0662 | 251.9091 |
| R² (val) | 0.3260 | 0.3085 | 0.3015 | 0.3464 |
| Pearson (val) | 0.5803 | 0.5604 | 0.5845 | 0.5885 |
| Custom Score (val) | 45.3125 | 43.4452 | 44.2989 | 46.7515 |
| Custom Score (test) | 34.9587 | 30.3308 | 33.8755 | 38.8637 |

- **RMSE**: root mean squared error on validation set.
- **R²**: coefficient of determination on validation set (negative values are clipped to 0 in our code).
- **Pearson**: absolute Pearson correlation between predictions and targets on validation set.
- **Custom Score**: composite score $100 \cdot \frac{R^2 + |\rho|}{2}$ where ρ is the Pearson correlation on the test set.

---

## 4 Future Work

Due to the fact that the dataset contains a small number of participants, the model receives data with high variance, so I consider moving the problem in frequence domain, because DFT act as also as a denoise method.

Considering Mouse for The Reading as an alternative for Eye Tracker, an spectral analysis to try to proove the equivalence between this two methods is an interesting study to be made.

---

## 5 Conclusion

This study is the proof that predicting total reading time it is not a simple task at all, due to the differences between participants, including varying cognitive processing speeds, disparities in educational backgrounds, and divergent levels of prior knowledge regarding the subject matter.

Crucially, this research also highlights that solving NLP problems requires a multi-disciplinary approach, integrating linguistic nuances with traditional machine learning.

---

## Limitations

The most important limitation of this study is the small number of participants in the dataset, which reduces the generalization ability of the models to a wider population of Romanian readers. In addition, all components are built specifically for Romanian, so applying this approach to other languages would require replacing the language-specific tools entirely. Finally, contextual feature extraction using RoGPT2 requires GPU resources and does not scale easily to larger datasets.

---

## References

HuggingFace Model Card. 2020. Rogpt2: Romanian gpt-2 (model). Hugging Face model page.

Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 785–794. ACM.

John Hale. 2001. A probabilistic earley parser as a psycholinguistic model. In *Proceedings of the Second Meeting of the North American Chapter of the Association for Computational Linguistics*, pages 1–8.

Anamaria Hodivoianu, Oleksandra Kuvshynova, Filip Popovici, Adrian Luca, and Sergiu Nisioi. 2025. Predicting total reading time using romanian eye-tracking data. In *Proceedings of the First International Workshop on Gaze Data and Natural Language Processing (Gaze4NLP 2025)*. Association for Computational Linguistics.

Matthew Honnibal and Ines Montani. 2017. spacy: Industrial-strength natural language processing in python. Software.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qi Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*.

Roger Levy. 2008. Expectation-based syntactic comprehension. *Cognition*, 106(3):1126–1177.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Andrey V Dorogush, and Anna Gulin. 2018. Catboost: unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems*, volume 31.