# IFT6758 - Data Science Project Repository

This repository contains the Final Project of the IFT6758 course of team members
- Fabrice Normandin
- Marie St-Laurent
- Rémi Dion
- Isabelle Viarouge

In this project, our objective was to identify the age, gender, and Big-5 personality traits of users using some (anonymized) data gathered from their facebook activity, including image, text, and page likes.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


## Prerequisites

What things you need to install the software and how to install them

First of all, you will require the following packages:
- `Tensorflow`
- `scikit-Learn`
- `pandas`
- `simple-parsing`: A self-authored python package used to simplify argument parsing, which is included in the `project/SimpleParsing` repository. (Can be installed with `pip install -e ./project/SimpleParsing`)
- `orion`: Hyperparameter tuning package from MILA)

These packages *should* be installed automatically by creating a new conda environment from the `conda environment.yml` file like so:
```bash
conda env create -f project/environment.yml
```

# Project Structure
```bash
└── project
    ├── baseline.py                 # contains the baseline implementation
    ├── environment.yml             # defines the 'datascience' conda environment used here.
    ├── hyperparameter_tuning.sh    # Used to launch HyperParameter tuning experiments with Orion
    ├── ift6758.py                  # contains an improved baseline using facial hair
    ├── model_old.py                # Contains an outdated model architecture as a backup
    ├── model.py                    # ** Contains the general (multi-head) Model code **
    ├── preprocessing_pipeline.py       # Preprocessing pipeline for both test and train
    ├── SimpleParsing                   # Python module to simplify argument parsing
    ├── show_server_training_plots.sh   # Script used to download and view experiment results
    ├── task_specific_models        # Contains backup (task-specific) models
    │   ├── age_group.py            # backup age_group model
    │   ├── gender.py               # backup gender model
    ├── test.py             # ** Test script invoked on server by ./ift6758 file **
    ├── train.py            # Training Script
    ├── user.py             # utility for describing a User as a dataclass
    ├── utils.py            # utility scripts
    |── workshop            
    └── workshop                # folder containing exploratory Jupyter notebooks
        ├── IsabelleWorkshop    # Jupyter notebooks of Isabelle Viarouge
        ├── Marie_tests         # Jupyter notebooks of Marie St-Laurent
        └── ws_rd               # Jupyter notebooks of Rémi Dion
```

# Training

To quickly launch a new training run with all default hyperparameters, use:
```bash
python ./project/train.py
```

The model structure is defined in the `model.py` script. The structure of the model can easily be changed by modifying any of the attributes of the `model.HyperParameters` class.

Under the current model architecture, each task (gender, age group, personality traits) share the same types of hyperparameters. Therefore, each task has a corresponding set of hyperparameters, represented as an instance of the `model.TaskHyperParameters` class, which can be found on the `gender`, `age_group`, and `personality` attributes of the `HyperParameters` class.




To see a list of all the possible HyperParameter values, call the `train.py` with the `--help` option, like so:

(Note: we use the [simple-parsing](https://github.com/lebrice/SimpleParsing) package to automatically create the all the following argparse arguments for us. Please contact Fabrice if interested).


```bash
$ python project/train.py --help
DEBUGGING:  True
usage: train.py [-h] [--batch_size int] [--activation str] [--optimizer str]
                [--learning_rate float] [--num_like_pages int]
                [--gender_loss_weight float] [--age_loss_weight float]
                [--max_number_of_likes int] [--embedding_dim int]
                [--shared_likes_embedding [str2bool]]
                [--use_custom_likes [str2bool]] [--gender.name str]
                [--gender.num_layers int] [--gender.num_units int]
                [--gender.activation str] [--gender.use_batchnorm [str2bool]]
                [--gender.use_dropout [str2bool]]
                [--gender.dropout_rate float]
                [--gender.use_image_features [str2bool]]
                [--gender.use_likes [str2bool]] [--gender.l1_reg float]
                [--gender.l2_reg float] [--gender.embed_likes [str2bool]]
                [--age_group.name str] [--age_group.num_layers int]
                [--age_group.num_units int] [--age_group.activation str]
                [--age_group.use_batchnorm [str2bool]]
                [--age_group.use_dropout [str2bool]]
                [--age_group.dropout_rate float]
                [--age_group.use_image_features [str2bool]]
                [--age_group.use_likes [str2bool]] [--age_group.l1_reg float]
                [--age_group.l2_reg float]
                [--age_group.embed_likes [str2bool]] [--personality.name str]
                [--personality.num_layers int] [--personality.num_units int]
                [--personality.activation str]
                [--personality.use_batchnorm [str2bool]]
                [--personality.use_dropout [str2bool]]
                [--personality.dropout_rate float]
                [--personality.use_image_features [str2bool]]
                [--personality.use_likes [str2bool]]
                [--personality.l1_reg float] [--personality.l2_reg float]
                [--personality.embed_likes [str2bool]] [--experiment_name str]
                [--log_dir str] [--validation_data_fraction float]
                [--epochs int] [--early_stopping_patience int]

optional arguments:
  -h, --help            show this help message and exit

HyperParameters ['hparams']:
  Hyperparameters of our model.

  --batch_size int      the batch size (default: 128)
  --activation str      the activation function used after each dense layer
                        (default: tanh)
  --optimizer str       Which optimizer to use during training. (default: sgd)
  --learning_rate float
                        Learning Rate (default: 0.001)
  --num_like_pages int  number of individual 'pages' that were kept during
                        preprocessing of the 'likes'. This corresponds to the
                        number of entries in the multi-hot like vector.
                        (default: 10000)
  --gender_loss_weight float
  --age_loss_weight float
  --max_number_of_likes int
  --embedding_dim int
  --shared_likes_embedding [str2bool]
  --use_custom_likes [str2bool]
                        Wether or not to use Rémis better kept like pages
                        (default: True)

TaskHyperParameters ['hparams.gender']:
  Gender model settings:

  --gender.name str     name of the task (default: gender)
  --gender.num_layers int
                        number of dense layers (default: 1)
  --gender.num_units int
                        units per layer (default: 32)
  --gender.activation str
                        activation function (default: tanh)
  --gender.use_batchnorm [str2bool]
                        wether or not to use batch normalization after each
                        dense layer (default: False)
  --gender.use_dropout [str2bool]
                        wether or not to use dropout after each dense layer
                        (default: True)
  --gender.dropout_rate float
                        the dropout rate (default: 0.1)
  --gender.use_image_features [str2bool]
                        wether or not image features should be used as input
                        (default: True)
  --gender.use_likes [str2bool]
                        wether or not 'likes' features should be used as input
                        (default: True)
  --gender.l1_reg float
                        L1 regularization coefficient (default: 0.005)
  --gender.l2_reg float
                        L2 regularization coefficient (default: 0.005)
  --gender.embed_likes [str2bool]
                        Wether or not a task-specific Embedding layer should
                        be used on the 'likes' features. When set to 'True',
                        it is expected that there no shared embedding is used.
                        (default: False)

TaskHyperParameters ['hparams.age_group']:
  Age Group Model settings:

  --age_group.name str  name of the task (default: age_group)
  --age_group.num_layers int
                        number of dense layers (default: 2)
  --age_group.num_units int
                        units per layer (default: 64)
  --age_group.activation str
                        activation function (default: tanh)
  --age_group.use_batchnorm [str2bool]
                        wether or not to use batch normalization after each
                        dense layer (default: False)
  --age_group.use_dropout [str2bool]
                        wether or not to use dropout after each dense layer
                        (default: True)
  --age_group.dropout_rate float
                        the dropout rate (default: 0.1)
  --age_group.use_image_features [str2bool]
                        wether or not image features should be used as input
                        (default: True)
  --age_group.use_likes [str2bool]
                        wether or not 'likes' features should be used as input
                        (default: True)
  --age_group.l1_reg float
                        L1 regularization coefficient (default: 0.005)
  --age_group.l2_reg float
                        L2 regularization coefficient (default: 0.005)
  --age_group.embed_likes [str2bool]
                        Wether or not a task-specific Embedding layer should
                        be used on the 'likes' features. When set to 'True',
                        it is expected that there no shared embedding is used.
                        (default: False)

TaskHyperParameters ['hparams.personality']:
  Personality Model(s) settings:

  --personality.name str
                        name of the task (default: personality)
  --personality.num_layers int
                        number of dense layers (default: 1)
  --personality.num_units int
                        units per layer (default: 8)
  --personality.activation str
                        activation function (default: tanh)
  --personality.use_batchnorm [str2bool]
                        wether or not to use batch normalization after each
                        dense layer (default: False)
  --personality.use_dropout [str2bool]
                        wether or not to use dropout after each dense layer
                        (default: True)
  --personality.dropout_rate float
                        the dropout rate (default: 0.1)
  --personality.use_image_features [str2bool]
                        wether or not image features should be used as input
                        (default: False)
  --personality.use_likes [str2bool]
                        wether or not 'likes' features should be used as input
                        (default: False)
  --personality.l1_reg float
                        L1 regularization coefficient (default: 0.005)
  --personality.l2_reg float
                        L2 regularization coefficient (default: 0.005)
  --personality.embed_likes [str2bool]
                        Wether or not a task-specific Embedding layer should
                        be used on the 'likes' features. When set to 'True',
                        it is expected that there no shared embedding is used.
                        (default: False)

TrainConfig ['train_config']:
  TrainConfig(experiment_name: str = 'debug', log_dir: str = '',
  validation_data_fraction: float = 0.2, epochs: int = 50,
  early_stopping_patience: int = 5)

  --experiment_name str
                        Name of the experiment (default: debug)
  --log_dir str         The directory where the model checkpoints, as well as
                        logs and event files should be saved at. (default: )
  --validation_data_fraction float
                        The fraction of all data corresponding to the
                        validation set. (default: 0.2)
  --epochs int          Number of passes through the dataset (default: 50)
  --early_stopping_patience int
                        Interrupt training if `val_loss` doesn't improving for
                        over `early_stopping_patience` epochs. (default: 5)

```

# Hyperparameter Tuning

To launch a new HyperParameter tuning experiment, call the `hyperparameter_tuning.sh` script, like so:

```bash
./project/hyperparameter_tuning.sh
```

This uses the `Orion` package to set different combinations of values to the arguments detailed above, following a given optimization algorithm. In our case, the algorithm is purely random.

The results of all previous experiments can easily be obtained and then viewed using the `show_server_training_plots.sh` script, like :
```bash
./project/show_server_training_plots.sh
```
This will rsync to download the experiment checkpoints into a local `server_checkpoints` folder, as well as the logs of all experiments into a local `server_logs` folder.

# Testing 

The `test.py` script is used to perform inference and construct the required `<userid>.xml` files.
Its arguments are detailed below:
```bash
python ./project/test.py --help
usage: test.py [-h] [--trained_model_dir TRAINED_MODEL_DIR] [-i I] [-o O]

optional arguments:
  -h, --help            show this help message and exit
  --trained_model_dir TRAINED_MODEL_DIR
                        directory of the trained model to use for inference.
  -i I                  Input directory
  -o O                  Output directory
```

You can use a specific model by providing the `--trained_model_dir` argument. When not provided, the default value is used, which corresponds to the `best_model_so_far` set in `model.py`.