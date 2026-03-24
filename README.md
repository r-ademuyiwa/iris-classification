# Iris Dataset Classifier
Iris classifier project.

## Description
This is a Machine Learning model for multiclass classification that uses 3 different types of classifiers
and scalers to label the flowers in the Iris dataset. 

The classifiers are KNearestNeighbours (KNN), Decision Trees and
Random Forrest.

The three scaling options are Standard, Robust and Minmax Scalers. These scaling are only possible when using the KNN
classifier.


## How to run
To run this project you would call uv using ```uv run```. Input the file path of the pyproject and by the file path of the dataset and 
then the type of classifier that the user wants to use in the command line.

If running KNN:
 ```bash
uv run .\Flower_Classification\Class_code.py .\Flower_Classification\iris.csv KNN  
````

After this, the program will ask the user to in put the type of scaler to be used.

### Development setup

Before we start developing, make sure you have the following tools installed:

Prerequisites:
- pipx

pipx is a tool to help you install and run Python applications in isolated environments. It is all we need to install the rest of the development tools.

1. Install pipx if you haven't already. You can find instructions [here](https://pipxproject.github.io/pipx/installation/).

2. Use pipx to install nox, uv (Universal Versioner), and pre-commit:
Note: The `[pbs]` extra for nox includes additional plugins, allowing it to download Python.

```bash
pipx install nox[pbs] uv pre-commit
```

3. Now, we can use nox to set up our development environment. Run the following commands in your terminal:

```bash
 nox -s install 
```

This command will create a virtual environment and install all the development dependencies specified in the `pyproject.toml` file. Populate the pyproject.toml file with your desired dependencies before running this command. *Note: You can run this command multiple times to ensure all dependencies are installed correctly.*

```bash

nox -s chores 
```

This command will run various code quality and formatting tools to ensure your code adheres to best practices.

```bash
 nox -s tests 
```

This command will run the test suite to ensure everything is working as expected.

```bash
 nox -s lock 
```

This command will update the lock files for your dependencies to ensure you have the latest compatible versions.


4. (Optional) (But recommended) Set up pre-commit hooks to automatically run code quality checks before each commit:

```bash
 pre-commit install 
```

This command will install the pre-commit hooks defined in the `.pre-commit-config.yaml` file. This helps maintain code quality by running checks before each commit.

That's it! You now have a fully set up development environment for your Python project. You can start coding, and the tools will help you maintain code quality and consistency. When you want to run Python scripts, please use:

```bash
uv run python ./path/to/myscript.py
```

   



