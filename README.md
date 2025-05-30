
# Classifying and Clustering Trading Agents

This project aims to classify and cluster different types of trading agents using various machine learning techniques, including Deep Neural Networks (DNN) and Support Vector Machines (SVM). The agents are characterized by various features derived from their trading behavior. The project includes data preprocessing, model training, evaluation, and explanation of the model's predictions. The work is part of research published in the article:

```bibtex
@article{wilinski2025classifying,
  title={Classifying and Clustering Trading Agents},
  author={Wilinski, Mateusz and Goel, Anubha and Iosifidis, Alexandros and Kanniainen, Juho},
  journal={arXiv preprint arXiv:2505.21662},
  year={2025}
}
```

If you find this repository useful, please cite the above article.

## Project Structure

```
cluster-investors/
├── configuration/
│   ├── dnn_tune_config.json
│   ├── svm_linear_grid.json
│   ├── svm_poly_grid.json
│   ├── svm_rbf_grid.json
├── data/
│   ├── x_train_<experiment>.csv
│   ├── y_train_<experiment>.csv
│   ├── x_test_<experiment>.csv
│   ├── y_test_<experiment>.csv
├── experiments/
│   ├── classification/
│   │   ├── run_dnn_classification.py
│   │   ├── run_svm_classification.py
│   │   ├── tune_dnn.py
│   │   ├── tune_svm.py
│   ├── clustering/
│   │   ├── optimal_clusters_hierarchical_clustering.py
│   │   ├── run_hierarchy_clustering.py
│   │   ├── stadion_scores_for_stable_method.py
├── plots/
│   ├── confusion/
│   │   └── dnn_cm_<experiment>_<n_features>.pdf
│   │   └── svm_cm_<experiment>_<n_features>.pdf
│   ├── explanation/
│   │   └── dnn_explain_<experiment>_<n_features>.pdf
│   │   └── svm_explain_<experiment>_<n_features>.pdf
└── README.md
```

## Requirements

- python 3.8+
- pytorch
- ray tune
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- kneed
- tslearn

You can install the required packages using pip:

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib ray kneed tslearn
```

## Running the Experiments

### Example for DNN Classification

To run the DNN classification experiment, navigate to the classification directory and execute the `run_dnn_classification.py` script:

```bash
python run_dnn_classification.py -experiment 1 -n_features 18 -epochs 20 -batch 32 -l1 256 -l2 1024 -l3 1024 -drop 0.2 -lr 1e-3 -plot_confusion True -print_report True -plot_explanation True
```

## License

This project is licensed under the MIT License.
