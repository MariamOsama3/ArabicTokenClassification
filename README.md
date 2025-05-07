# Arabic POS Tagging with Transformers

<MariamOsama3/Mariam_classifer2>
A simple, end-to-end pipeline for tagging Arabic text with Part-Of-Speech (POS) labels using a transformer model.

---

After training, the model achieved:

* **Loss:** 0.0604 (training), 0.0874 (validation)
* **Precision:** 0.9670
* **Recall:** 0.9651
* **F1 Score:** 0.9660
* **Accuracy:** 0.9752

---

## Dataset

The dataset consists of Arabic sentences in CoNLL-U format, tagged with Universal Dependencies POS labels. Place your `.conllu` file in the `data/` folder.

---

## How to Use

1. **Install Requirements**: Ensure you have Python 3.8+, TensorFlow, and the Transformers library.
2. **Prepare Data**: Put your Arabic POS file in `data/`.
3. **Train Model**: Run the training script or notebook—this will load data, train the model, and save results.
4. **Evaluate**: After training, view the reported metrics above or evaluate on a test set.
5. **Inference**: Load the saved model and tokenizer to tag new Arabic text.

---

## Contributing

Feel free to suggest improvements by opening an issue or pull request.

--
