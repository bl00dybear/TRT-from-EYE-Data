<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Total Reading Time Prediction from Eye-Tracking Data</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #c8c5c0;
    font-family: 'Georgia', 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #111;
    padding: 30px 0 40px 0;
  }

  .page {
    width: 210mm;
    min-height: 297mm;
    margin: 0 auto;
    background: #fff;
    padding: 20mm 18mm 22mm 18mm;
    position: relative;
  }

  /* ── Page separator ── */
  .page-break {
    width: 210mm;
    margin: 0 auto;
    height: 32px;
    background: #c8c5c0;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .page-break::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0; right: 0;
    height: 1px;
    background: #999;
  }
  .page-break span {
    background: #c8c5c0;
    padding: 0 12px;
    font-size: 7.5pt;
    font-family: 'Georgia', serif;
    color: #666;
    position: relative;
    z-index: 1;
    letter-spacing: 0.08em;
    font-style: italic;
  }

  /* ── Title ── */
  .title-block {
    text-align: center;
    margin-bottom: 14px;
  }
  .title-block h1 {
    font-size: 15pt;
    font-weight: bold;
    line-height: 1.25;
    margin-bottom: 9px;
  }
  .authors { font-size: 11pt; margin-bottom: 2px; }
  .email {
    font-size: 8.5pt;
    color: #333;
    font-family: 'Courier New', monospace;
  }

  /* ── Abstract ── */
  .abstract-block {
    margin: 10px 12mm 14px 12mm;
  }
  .abstract-block .abs-title {
    font-size: 9.5pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
    font-variant: small-caps;
    letter-spacing: 0.06em;
  }
  .abstract-block p {
    font-size: 9pt;
    text-align: justify;
    hyphens: auto;
  }

  hr.divider {
    border: none;
    border-top: 1px solid #444;
    margin: 10px 0 12px 0;
  }

  /* ── Two-column ── */
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    column-gap: 14px;
  }
  .full-width { grid-column: 1 / -1; }

  /* ── Headings ── */
  h2.sec {
    font-size: 10pt;
    font-weight: bold;
    margin: 11px 0 4px 0;
    font-variant: small-caps;
    letter-spacing: 0.05em;
  }
  h3.subsec {
    font-size: 10pt;
    font-weight: bold;
    margin: 8px 0 3px 0;
  }
  h4.subsubsec {
    font-size: 10pt;
    font-style: italic;
    font-weight: normal;
    margin: 7px 0 2px 0;
  }

  p {
    text-align: justify;
    hyphens: auto;
    margin-bottom: 5px;
  }

  code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt;
    background: #f3f3f3;
    padding: 0 2px;
    border-radius: 2px;
  }

  /* ── Table ── */
  .table-wrap {
    grid-column: 1 / -1;
    margin: 10px 0 8px 0;
  }
  .table-caption {
    font-size: 9pt;
    text-align: center;
    margin-bottom: 5px;
    font-weight: bold;
  }
  table.results {
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
  }
  table.results thead tr {
    border-top: 1.5px solid #111;
    border-bottom: 1px solid #111;
  }
  table.results tbody tr:last-child {
    border-bottom: 1.5px solid #111;
  }
  table.results th, table.results td {
    padding: 3px 6px;
    text-align: center;
  }
  table.results th:first-child,
  table.results td:first-child { text-align: left; }

  .formula {
    text-align: center;
    margin: 7px 0;
  }

  ul.notes {
    font-size: 9pt;
    list-style: none;
    padding-left: 0;
    margin: 4px 0 6px 0;
  }
  ul.notes li { margin-bottom: 2px; padding-left: 1em; text-indent: -1em; }
  ul.notes li::before { content: "– "; }

  .references p {
    font-size: 8.2pt;
    text-indent: -1.5em;
    padding-left: 1.5em;
    margin-bottom: 4px;
    text-align: justify;
  }

  .page-num {
    position: absolute;
    bottom: 10mm;
    left: 0; right: 0;
    text-align: center;
    font-size: 9pt;
    color: #555;
  }

  @media print {
    body { background: white; padding: 0; }
    .page { margin: 0; box-shadow: none; width: 100%; page-break-after: always; min-height: unset; }
    .page-break { display: none; }
  }
</style>
</head>
<body>

<!-- ══════════════════════ PAGE 1 ══════════════════════ -->
<div class="page">

  <div class="title-block">
    <h1>Total Reading Time Prediction from Eye-Tracking Data</h1>
    <div class="authors">S,incari Sebastian George</div>
    <div class="email">sebastian-george.sincari@s.unibuc.ro</div>
  </div>

  <div class="abstract-block">
    <div class="abs-title">Abstract</div>
    <p>This study investigates the prediction of word-level Total Reading Time from eye-tracking data collected on Romanian texts. The dataset contains only the word, its position index, the text ID, and the total reading time, so a preprocessing pipeline is built to extract lexical, morpho-syntactic, and contextual features. Four models are evaluated: a LightGBM regressor, a Hybrid LightGBM, a FeedForward Neural Network, and a Stacked Ensemble of gradient boosting models. The results show that the Stacked Ensemble achieves the best performance, with a custom score of 38.86 on the test set, confirming that combining multiple models produces more robust predictions than any single approach.</p>
  </div>

  <hr class="divider">

  <div class="two-col">

    <!-- LEFT -->
    <div>
      <h2 class="sec">1&nbsp;&nbsp;Introduction</h2>
      <p>Reading time prediction is the problem of estimating how long a person looks at a word while reading a sentence. This is not a simple task, beacause reading speed is influenced by many factors at the same time, such as word frequency, syntactic complexity, and the predictability of a word given its context. Understanding these factors is important for natural language processing, because it connects the way models process text with the way humans actually read it.</p>
      <p>This study was developed as a semester project, but the subject is treated as a personal research problem. The reason is that the combination of cognitive signals and machine learning is, in my opinion, one of the most interesting directions in modern NLP. Eye-tracking data gives a direct measurement of human reading behavior, and the challenge of predicting it from linguistic features alone is both technically difficult and scientifically meaningful.</p>
      <p>The specific task is to predict word-level Total Reading Time for Romanian texts, using a dataset collected with an Eye Tracker. This is a particularly difficult setting, beacause Romanian is a morphologically rich language with flexible word order, and public cognitive corpora for Romanian did not exist until very recently (Hodivoianu et al., 2025).</p>
      <p>The preprocessing pipeline, the models, and the evaluation are described in detail in Section 3. The results and a discussion of their implications are presented in Section ??.</p>

      <h2 class="sec">2&nbsp;&nbsp;Related Work</h2>
      <p>Eye-tracking data has become a useful tool in natural language processing, because it shows how humans read and process text. Most early work on predicting reading times used simple regression models with handcrafted features like word frequency and word length. However, this type of research for the Romanian language was very limited, beacause no public eye-tracking dataset existed.</p>
      <p>This changed with (Hodivoianu et al., 2025), who released the first Romanian eye-tracking dataset for reading and established the current state-of-the-art for predicting word-level Total Reading Time. Their study showed that combining linguistic features with contextual surprisal inside ensemble models produces strong results, even when compared to more complex neural approaches.</p>

      <h2 class="sec">3&nbsp;&nbsp;Methodology</h2>
      <h3 class="subsec">3.1&nbsp;&nbsp;Dataset</h3>
      <p>The dataset used in this study was collected using an Eye Tracker. The subjects of this experiment had to read ten, plus two for accomodation with the setup. As an approval for data that was collected from them, they had to respond at six questions after each text. This ensures that the data are valid and reflect genuine, attentive reading, confirming that participants were engaged and did not merely simulate reading.</p>
      <p>However, the dataset used in this study does not include raw eye-tracking metrics; it contains only the word, its position index on the page, the text ID, and the total reading time. In this case, the preprocessing step in which to extract more features is absolutly necessary.</p>

      <h3 class="subsec">3.2&nbsp;&nbsp;Preprocessing</h3>
      <p>The features extracted from the dataset used in this study can be categorized in following classes: lexical, morpho-syntactic and contextual features.</p>

      <h4 class="subsubsec">3.2.1&nbsp;&nbsp;Lexical features</h4>
      <p>Lexical feature extraction process is fairly direct, extracting word length (including punctuation), squared word length ment to model non-linear effects, number of diacritics (specific of Romanian words), binary indicator if it is capitalized, binary indicator for sentence-initial position, binary indicator for punctuation at the end of the word, binary flag for presence of non-alphabetic characters. In addition, other extracted lexicat feature, based on Zipf scale, word frequency score in Romanian; using spaCy (Honnibal and Montani, 2017) a binary indicator for Romanian stopword, number of syllables count using a heuristic vowel-group method (count transitions from non-vowel to vowel in "aeiouăâî") and a vowel-consonant ratio ((# vowels) / (word length)).</p>
    </div>

    <!-- RIGHT -->
    <div>
      <h4 class="subsubsec">3.2.2&nbsp;&nbsp;Morpho-Syntactic Features</h4>
      <p>Morpho-syntactic features are extracted using spaCy (Honnibal and Montani, 2017) after reconstructing the text. Each word is assigned a part-of-speech tag (spaCy) and a dependency relation (spaCy), both encoded as numeric values. The dependency depth is also computed (spaCy) as the number of ancestors in the syntax tree, used as a simple measure of syntactic complexity.</p>
      <p>A binary indicator for content words is included, based on the tags <code>NOUN</code>, <code>VERB</code>, <code>ADJ</code>, <code>ADV</code>, <code>PROPN</code> (spaCy). From named entity recognition, a binary feature shows if a word is part of a named entity, together with its type (spaCy).</p>
      <p>Morphological features include the number of attributes (spaCy), grammatical case (encoded as <code>Nom</code>, <code>Acc</code>, <code>Dat</code>, <code>Gen</code>, <code>Voc</code>, extracted with spaCy), and verb form (encoded as <code>Fin</code>, <code>Inf</code>, <code>Part</code>, <code>Ger</code>, <code>Sup</code>, also from spaCy). A binary feature also indicates if the word is inflected (comparison between token and lemma, spaCy).</p>

      <h4 class="subsubsec">3.2.3&nbsp;&nbsp;Contextual Features</h4>
      <p>Contextual features are computed using a pretrained Romanian language model, RoGPT2 (Card, 2020) (via <code>transformers</code>). The text is tokenized (<code>transformers</code> tokenizer) and processed in chunks. For each word, a surprisal score is calculated as the negative log-probability (base 2) of its tokens given the context (RoGPT2).</p>
      <p>This surprisal value reflects how predictable a word is in context and is linked to reading difficulty (Hale, 2001; Levy, 2008).</p>
      <p>Parafoveal features are also included by shifting values within each text and page (<code>pandas</code>). These include previous and next word surprisal (from RoGPT2), length (basic string processing), and frequency (from <code>wordfreq</code>), capturing local context effects.</p>
      <p>At a higher level, a relative position feature is computed as the word index divided by total words in the page (<code>pandas</code>). A type-token ratio is calculated per text using grouping operations (<code>pandas</code>) to measure lexical diversity. Finally, an interaction between surprisal and frequency is added: <code>surprisal * log2(1 + frequency)</code> (NumPy).</p>

      <h3 class="subsec">3.3&nbsp;&nbsp;Method</h3>
      <p>The prediction of reading time is formulated as a regression task, where the target variable is the total reading time per word, as recorded by the Eye Tracker. Four models are evaluated in this study: a LightGBM regressor, a Hybrid LightGBM, a Feed-Forward Neural Network, and a Stacked Ensemble.</p>

      <h4 class="subsubsec">3.3.1&nbsp;&nbsp;LightGBM</h4>
      <p>LightGBM is a gradient boosting method that builds decision trees sequentially, where each new tree corrects the errors of the previous one. This makes it well suited for tabular data with mixed feature types, such as the combination of continuous linguistic scores and categorical morpho-syntactic tags used in this study. (Ke et al., 2017)</p>
      <p>The feature matrix contains lexical and linguistic numeric features (for example <code>word_len</code>, <code>word_len_sq</code>, <code>n_diacritics</code>, <code>zipf_frequency</code>, <code>n_syllables</code>, <code>vowel_consonant_ratio</code>, <code>is_stopword</code>, <code>is_NE</code>, <code>is_content_word</code>, <code>contextual_surprisal</code>, <code>morph_count</code>, <code>is_inflected</code>). To extend the contextual window available to the model, lag and lead features are computed for <code>word_len</code>, <code>contextual_surprisal</code>, and <code>zipf_frequency</code> at offsets t−1, t−2, t+1, and t+2 within each <code>participant_id</code>/<code>text</code> group, capturing the effect of neighboring words on the reading time of the current one.</p>
      <p>Validation uses a participant-level split, meaning the model is evaluated on unseen readers, not unseen words. The model is trained with hyperparameters <code>num_leaves</code>, <code>learning_rate</code>, <code>min_child_samples</code>, <code>subsample</code>, <code>colsample_bytree</code>, <code>reg_alpha</code>, and <code>reg_lambda</code>, tuned manually. Early stopping on the validation set is applied to find the optimal number of trees and prevent overfitting.</p>
    </div>

  </div>
  <div class="page-num">1</div>
</div>

<!-- ══════════════════════ SEPARATOR ══════════════════════ -->
<div class="page-break"><span>page 2</span></div>

<!-- ══════════════════════ PAGE 2 ══════════════════════ -->
<div class="page">
  <div class="two-col">

    <!-- LEFT -->
    <div>
      <h4 class="subsubsec">3.3.2&nbsp;&nbsp;Hybrid LightGBM</h4>
      <p>The Hybrid LightGBM pipeline extends the standard approach by splitting the prediction into two stages. First, a binary classifier predicts whether the reading time is greater than zero, using <code>answer &gt; 0</code> as the label. Second, a regression model is trained only on samples with positive reading time values. At inference time, the final output is the product of the classifier prediction, after thresholding at 0.5, and the regressor output. In other words, if the classifier decides a word was not fixated, the predicted reading time is zero. The same feature preparation, lag and lead features, and participant-level validation split are used in both stages.</p>

      <h4 class="subsubsec">3.3.3&nbsp;&nbsp;Feed-Forward Neural Network</h4>
      <p>The Feed-Forward Neural Network is a tabular regression model built as a multilayer perceptron (a stack of linear transformations followed by non-linear activations, allowing the model to capture complex interactions between features). It uses only numeric features, standardized using the mean and standard deviation from the training set.</p>
      <p>The architecture has three hidden layers with sizes 512, 256, and 128 neurons. Each hidden layer is followed by batch normalization (which stabilizes training by normalizing the activations), a ReLU activation, and dropout regularization to prevent co-adaptation of neurons. The output layer is a single linear neuron, appropriate for regression.</p>
      <p>The target is transformed with <code>log1p</code> during training and converted back with <code>expm1</code> at inference time, which reduces the effect of large outliers in reading time. The model is trained with the Adam optimizer and Huber loss (a loss function that is less sensitive to outliers than mean squared error). A learning-rate scheduler reduces the learning rate when validation performance stops improving, and training stops early when no improvement is observed for several consecutive epochs.</p>

      <h4 class="subsubsec">3.3.4&nbsp;&nbsp;Stacked Ensemble</h4>
      <p>The Stacked Ensemble combines three gradient boosting regressors: CatBoost (Prokhorenkova et al., 2018), LightGBM (Ke et al., 2017), and XGBoost (Chen and Guestrin, 2016). Each base model is trained using GroupKFold cross-validation with <code>participant_id</code> as the grouping variable, ensuring that each validation fold contains only unseen participants. The same feature preparation, including lag and lead features and categorical encoding, is applied to all three models.</p>
      <p>For each fold, the three base models produce out-of-fold predictions on the validation participants. These predictions are stacked into a new feature matrix, which is used to train a Ridge meta-model (a linear regression with L2 regularization that learns the optimal weight for each base model's contribution). The final prediction is the output of this meta-model. For the test set, predictions from each base model are averaged across all folds before being passed to the Ridge regressor.</p>

      <h4 class="subsubsec">3.3.5&nbsp;&nbsp;Evaluation</h4>
      <p>Model performance is evaluated using four metrics. RMSE (Root Mean Squared Error) measures the average magnitude of prediction errors in the same unit as reading time. R² measures the proportion of variance in reading time explained by the model. Pearson correlation measures the linear agreement between predicted and actual values. Finally, a composite score is computed as:</p>
      <div class="formula">
        \[ \text{Score} = 100 \cdot \frac{R^2 + |\rho|}{2} \]
      </div>
      <p>where ρ is the Pearson correlation. This score balances explained variance with rank-order agreement and serves as the primary evaluation criterion.</p>
    </div>

    <!-- RIGHT -->
    <div>
      <h4 class="subsubsec">3.3.6&nbsp;&nbsp;Results</h4>
      <p>The table below summarizes validation metrics for each model and the custom score on the test set. Validation metrics are computed on the held-out participant validation set. See Table 1 for the full results.</p>
      <ul class="notes">
        <li>RMSE: root mean squared error on validation set.</li>
        <li>R²: coefficient of determination on validation set (negative values are clipped to 0 in our code).</li>
        <li>Pearson: absolute Pearson correlation between predictions and targets on validation set.</li>
        <li>Custom Score: \( 100 \cdot \frac{R^2 + |\rho|}{2} \) where ρ is the Pearson correlation on the test set.</li>
      </ul>

      <h2 class="sec">4&nbsp;&nbsp;Future Work</h2>
      <p>Due to the fact that the dataset contains a small number of participants, the model receives data with high variance, so I consider moving the problem in frequence domain, because DFT act as also as a denoise method.</p>
      <p>Considering Mouse for The Reading as an alternative for Eye Tracker, an spectral analysis to try to proove the equivalence between this two methods is an interesting study to be made.</p>

      <h2 class="sec">5&nbsp;&nbsp;Conclusion</h2>
      <p>This study is the proof that predicting total reading time it is not a simple task at all, due to the differences between participants, including varying cognitive processing speeds, disparities in educational backgrounds, and divergent levels of prior knowledge regarding the subject matter.</p>
      <p>Crucially, this research also highlights that solving NLP problems requires a multi-disciplinary approach, integrating linguistic nuances with traditional machine learning.</p>

      <h2 class="sec">Limitations</h2>
      <p>The most important limitation of this study is the small number of participants in the dataset, which reduces the generalization ability of the models to a wider population of Romanian readers. In addition, all components are built specifically for Romanian, so applying this approach to other languages would require replacing the language-specific tools entirely. Finally, contextual feature extraction using RoGPT2 requires GPU resources and does not scale easily to larger datasets.</p>
    </div>

    <!-- Table: full width -->
    <div class="table-wrap">
      <div class="table-caption">Table 1: Validation metrics and final test score</div>
      <table class="results">
        <thead>
          <tr>
            <th>Metric</th>
            <th>LightGBM</th>
            <th>Hybrid LightGBM</th>
            <th>Neural Network</th>
            <th>Stacked Ensemble</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>RMSE (val)</td><td>285.9981</td><td>289.6788</td><td>233.0662</td><td>251.9091</td></tr>
          <tr><td>R² (val)</td><td>0.3260</td><td>0.3085</td><td>0.3015</td><td>0.3464</td></tr>
          <tr><td>Pearson (val)</td><td>0.5803</td><td>0.5604</td><td>0.5845</td><td>0.5885</td></tr>
          <tr><td>Custom Score (val)</td><td>45.3125</td><td>43.4452</td><td>44.2989</td><td>46.7515</td></tr>
          <tr><td>Custom Score (test)</td><td>34.9587</td><td>30.3308</td><td>33.8755</td><td>38.8637</td></tr>
        </tbody>
      </table>
    </div>

    <!-- References: full width -->
    <div class="full-width">
      <h2 class="sec">References</h2>
      <div class="references">
        <p>HuggingFace Model Card. 2020. Rogpt2: Romanian gpt-2 (model). Hugging Face model page.</p>
        <p>Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In <em>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</em>, pages 785–794. ACM.</p>
        <p>John Hale. 2001. A probabilistic earley parser as a psycholinguistic model. In <em>Proceedings of the Second Meeting of the North American Chapter of the Association for Computational Linguistics</em>, pages 1–8.</p>
        <p>Anamaria Hodivoianu, Oleksandra Kuvshynova, Filip Popovici, Adrian Luca, and Sergiu Nisioi. 2025. Predicting total reading time using romanian eye-tracking data. In <em>Proceedings of the First International Workshop on Gaze Data and Natural Language Processing (Gaze4NLP 2025)</em>. Association for Computational Linguistics.</p>
        <p>Matthew Honnibal and Ines Montani. 2017. spacy: Industrial-strength natural language processing in python. Software.</p>
        <p>Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qi Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly efficient gradient boosting decision tree. In <em>Advances in Neural Information Processing Systems 30</em>.</p>
        <p>Roger Levy. 2008. Expectation-based syntactic comprehension. <em>Cognition</em>, 106(3):1126–1177.</p>
        <p>Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Andrey V Dorogush, and Anna Gulin. 2018. Catboost: unbiased boosting with categorical features. In <em>Advances in Neural Information Processing Systems</em>, volume 31.</p>
      </div>
    </div>

  </div>
  <div class="page-num">2</div>
</div>

</body>
</html>