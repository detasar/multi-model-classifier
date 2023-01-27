# Multi-Model ML Classifier

This repository contains a script that pre-processes an input CSV file, separates the target column, performs train-test split, and fits multiple machine learning models with different hyperparameters. The results are then written to a dataframe and the best model is picked and returned.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need to have Python and the following libraries installed:
- pandas
- numpy
- sklearn


### Installing

Clone the repository to your local machine using the following command:

git clone https://github.com/detasar/multi-model-classifier.git


## Running the script

1. Place the input CSV file in the same directory as the script.
2. In the `main.py` file, update the `filepath` variable to the name of the input CSV file and the `target_col` variable to the name of the target column in the input file.
3. Run the script by executing the following command:

python main.py


## Results

The script will print the best model and the results dataframe containing the accuracy of each model.

## Customization

You can add more models, more preprocessing methods or any other functionality as you see fit.

## Built With

* [Python](https://www.python.org/)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/)

## Author

* **EMRE TASAR** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
