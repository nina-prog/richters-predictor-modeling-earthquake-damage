# Data Science Lab 2023: Group 5 Targaryen 
This repository contains our project of the phase 1 of the **Practical Course: Data Science for Scientific Data** at Karlsruhe Institute of Technology (KIT). The project is about the 'Richter's Predictor: Modeling Earthquake Damage' competition ([Link](https://www.drivendata.org/competitions/57/)).

## Group Members: 
| Forename | Surname  | Matr.#  |
|----------|----------|---------|
| Nina     | Mertins  | 2107539 |
| Kevin    | Hartmann | 1996265 |
| Alessio  | Negrini  | 2106547 |

## Folder Structure
```
📦phase-1
 ┣ 📂config                  <-- Configuration files for the pipeline
 ┣ 📂data                    <-- Data used as input during development with Jupyter notebooks. 
 ┃ ┣ 📂raw                   <-- Contains the raw data provided by the supervisors.
 ┃ ┗ 📂processed             <-- Contains the processed data build during development.
 ┣ 📂models                  <-- Saved models during Development.
 ┣ 📂notebooks               <-- Jupyter Notebooks used in development.
 ┃ ┗ 📂weekXX                <-- Contains the notebooks for weekly subtasks and experimenting.
 ┣ 📂src                     <-- Source code.
 ┣ 📜.gitignore 
 ┣ 📜README.md               <-- The top-level README for developers using this project. 
 ┗ 📜requirements.txt        <-- The requirenments file for reproducing the environment, e.g. generated with 
                                 'pip freeze > requirenments.txt'.
```

## Setting up the environment and run the code on a local machine
Note: Make sure Python 3.10 is installed

1. Clone the repository with:

       git clone https://git.scc.kit.edu/data-science-lab-2023/group-5-targaryen/phase-1.git

2. Install the requirements:

       pip install -r phase-1/requirements.txt

3. Insert the data in phase-1/data/raw
4. Navigate into project root and execute the pipeline with:

       cd phase-1
       python3 src/main.py --config "configs/config.yml"

## Execute on a linux server using a script
The build.sh script includes the correct command to run the Python code, and to specifies any required command line 
arguments or configuration files.

1. Clone the GitLab repository to your server using the git clone command. For example:

```git clone https://git.scc.kit.edu/data-science-lab-2023/group-5-targaryen/phase-1.git```

2. Change into the cloned repository directory:

```cd phase-1```

3. Make the build.sh script executable:

```chmod +x build.sh```

4. Run the build.sh script to install any required dependencies and run the Python code:

```./build.sh```

This should execute the script and run the Python code specified in src/main.py.

That's it! With these steps, you should be able to execute the Python code from this Repo on your Linux server.
