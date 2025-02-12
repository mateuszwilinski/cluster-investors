
# Classifying and Clustering Trading Agents

This project aims to classify and cluster different types of trading agents using various machine learning techniques, including Deep Neural Networks (DNN) and Support Vector Machines (SVM). The agents are characterized by various features derived from their trading behavior. The project includes data preprocessing, model training, evaluation, and explanation of the model's predictions. The work is part of research published in the article:

```bibtex
@article{wilinski2025classifying,
  title={Classifying and Clustering Trading Agents},
  author={Wilinski, Mateusz and Goel, Anubha and Iosifidis, Alexandros and Kanniainen, Juho},
}
```

If you find this repository useful, please cite the above article.

## Project Structure

```
cluster-investors/
├── data/
│   ├── x_train_<experiment>.csv
│   ├── y_train_<experiment>.csv
│   ├── x_test_<experiment>.csv
│   ├── y_test_<experiment>.csv
├── experiments/
│   ├── classification/
│   │   ├── run_dnn_classification.py
│   │   ├── run_svm_classification.py
│   ├── hyperparameters/
│   │   ├── dnn_hyperparameters.py
│   │   ├── svm_hyperparameters.py
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

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- Matplotlib

You can install the required packages using pip:

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

## Running the Experiments

### DNN Classification

To run the DNN classification experiment, navigate to the classification directory and execute the `run_dnn_classification.py` script:

```bash
python run_dnn_classification.py -experiment <experiment_number> -n_features <number_of_features> -epochs <number_of_epochs> -batch <batch_size> -l1 <layer1_size> -l2 <layer2_size> -l3 <layer3_size> -drop <dropout_rate> -lr <learning_rate> -plot_confusion <True/False> -print_report <True/False> -plot_explanation <True/False>
```

### SVM Classification

To run the SVM classification experiment, navigate to the classification directory and execute the `run_svm_classification.py` script:

```bash
python run_svm_classification.py -experiment <experiment_number> -n_features <number_of_features> -kernel <kernel_type> -C <regularization_parameter> -gamma <gamma_value> -plot_confusion <True/False> -print_report <True/False> -plot_explanation <True/False>
```

### Example

```bash
python run_dnn_classification.py -experiment 1 -n_features 18 -epochs 20 -batch 32 -l1 256 -l2 1024 -l3 1024 -drop 0.2 -lr 1e-3 -plot_confusion True -print_report True -plot_explanation True
```

```bash
python run_svm_classification.py -experiment 1 -n_features 18 -kernel rbf -C 1.0 -gamma scale -plot_confusion True -print_report True -plot_explanation True
```

## Script Overview

### `run_dnn_classification.py` and `run_svm_classification.py`

These scripts perform training and evaluation of given models, DNN and SVM respectively. They also perform Layer-Wise Relevance Propagation for the trained DNN, and compute the SVM one-vs-one weights, which are used to explain how the methods managed to classify agents. Finally, all the results are visualised.

### `dnn_hyperparameters.py` and `svm_hyperparameters.py`

These scripts are used for hyperparameter tuning of the DNN and SVM models, respectively. They use techniques such as grid search or random search to find the best hyperparameters for the models.

## Results

The results of the experiments, including the confusion matrix and explanation plots, are saved in the plots directory.

## License

This project is licensed under the MIT License.
