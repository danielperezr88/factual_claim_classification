# Factual claim classification with a BERT-based fine-tuned model

**Note**: Training data is not added to this repository on purpose, as I didn't have ownership nor
permission to share it. However, the data model can be easily figured out by checking
[the Markdown report](factual_claim_classification.md), and a quick search on Google makes easy to
find even bigger public factual claim classification datasets.

All code for this example resides on [the notebook](factual_claim_classification.ipynb), of which
you may also find a Markdown [report for a complete successful run](factual_claim_classification.md).

Although the notebook is thoroughtly commented, I'll share a brief summary here:
- The notebook is designed to be **compatible with Google Colab's TPUs**, although in practice free
  tier TPUs have shown worse performance than GPUs. The reason for this could be times spent moving
  data to/from those and the fact that training data for this exercise is very small.
- In the first cells **there's code for cloning the repo in Google Colab and downloading from a
  public Google Storage bucket the model files** needed to obtain the results shown in
  [the Markdown report](factual_claim_classification.md). Code for cloning the repo asks for
  credentials, so you must have a GitHub account to use that.
- The model chosen for fine tuning is **Google's recommended version of Multilingual BERT**, called
  'bert-base-multilingual-cased' (non-case-agnostic version), from which I've decided to **freeze all
  layers except for the classifier layers and the last fully connected layer** I've added to it.
- Regarding training data preparation, I've decided to **fix class imbalance in training data** through
  upsampling of the minority class. Also, as samples came in groups of 3, I had to rearrange them.
  The rest of effort in data preparation was put in a custom **PyTorch Dataset class for better
  controlled tokenization**.
- For evaluation of results I decided to go with **Precision / Recall curve**, using both its AUC and
  plotted figure as a basis to **select best performing models** to save during training epochs and to
  **find the optimal cutoff points** at the end of the exercise under different scenarios.
- Finally, I've presented a **classification metrics report** which incorporates some close-to-customer
  figures which would smooth the presentation of results for any non-expert audience.
- The document ends with a listing of potential **next steps** of interest.

Enjoy :)
