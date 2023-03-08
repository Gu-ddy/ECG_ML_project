# AML task 2
## About this repo
This repository was created by myself to collaborate on a project from a Master's course in Advanced Machine Learning. Below is a list explaining the relevant folders, as well as a set of instructions I wrote up for my colleagues, who weren't familiar with GitHub, on how to connect it to their IDEs in an easy way, as well as some general best practices. 

In this project, the aim was to classify patients into one of 4 classes (what the classes represented was unknown) based on raw electrocardiogram readings. What was given to us for this project was raw training data and raw test data, which we cleaned and used to train several models and make our final predictions.

This repo contains the following folders, with their corresponding content:
- **Euler**: a markdown file and a screenshot explaining how to connect to university clusters and how to use the batch system to speed up training. 
- **Models**: contains programs to train and test models on our processed data. The models that yielded the best results are in stack.py, which trained a stacking classifier with a bag of automatically selected models, and voting.py, which trained a voting classifier with a bag of automatically selected models.
- **feature_extraction**: one program to extract as many features as possible from the raw electrocardiogram data. Due to the lack of clarity on our initial approach, this program is not optimized for what was ultimately its goal, yet it achieved the results we set out to attain.
- **heartbeat_extraction**: several attempts at using different python libraries to extract heartbeat information, such as peak locations and peak types, from the raw electrocardiogram dataset.

## Best practices
### Connecting your IDE
General best practices/things to do: 
- connect your github account to PyCharm/your IDE (
  https://www.youtube.com/watch?v=KEfo6sHgVOc&ab_channel=ecodedev from 1:18 to about 4:28)
- the "master" branch is called main
- whenever you want to add/modify code:
    - update the main branch (on PyCharm: click on "main" in bottom right of your screen, click on main again and select update)
    - checkout from the main branch with branch name development_<your_name>
    - make your changes
    - commit your changes (in PyCharm: green check mark top right)
        - it will ask you to write a comment explaining your commit
        - then click commit or "commit and push" is generally easier
    - add a request to pull your branch onto main (can be done from GitHub website or from command line probably but idk how)
    - merge the pull request onto main (idk yet if only I can do that or you guys can too)
    
    
- The next time you want to add/modify code, you can repeat the same (checking out onto a new branch from main) or you can 
  - select the development branch with your name and checkout onto it
  - update main 
  - from the "main" dropdown list, select "merge into current" 
  - then proceed to do your modifications
    
Additional info on how it works: https://medium.com/@androidmatheny/using-git-and-github-on-group-projects-d636be2cdd4d

Try to comment your code and name things so that they're understandable, e.g. X_train_no_outliers vs X_WoO. Document each file 
and function saying what is expected as input and output, i.e. data type, sizes, ...

If you feel like it, I recommend installing the Black tool for PyCharm: https://black.readthedocs.io/en/stable/integrations/editors.html
It cleans the code and makes it more legible

### requirements.txt
you'll notice there is a file named requirements.txt . This file is to ensure we all know
what libraries are needed to run the code and so that we can install these libraries easily 
with the command

`pip install -r requirements.txt`
