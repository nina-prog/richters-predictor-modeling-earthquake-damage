# Data Science Lab 2023: Group 5 Targaryen 

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
 ┃ ┗ 📂raw                   <-- Contains the raw data provided by the supervisors.
 ┣ 📂models                  <-- Saved models during Development.
 ┣ 📂notebooks               <-- Jupyter Notebooks used in development.
 ┃ ┗ 📂weekXX                <-- Contains the weekly subtasks.
 ┣ 📂src                     <-- The customized project packages containing all utility functions and source codes.
 ┣ 📜.gitignore 
 ┣ 📜README.md               <-- The top-level README for developers using this project. 
 ┗ 📜requirements.txt        <-- The requirenments file for reproducing the environment, e.g. generated with 
                                 'pip freeze > requirenments.txt'.
```

## Setting up the environment and run the code
1. Clone the repository with:  

       git clone git@git.scc.kit.edu:data-science-lab-2023/group-5-targaryen/phase-1.git

2. Install the requirements:

       pip install -r pahse-1/requirements.txt

3. Insert the data in phase-1/data/raw
4. Navigate in the source folder and execute the pipeline with:

       cd phase-1/src
       python main.py --config "../configs/sample_config.yml"
