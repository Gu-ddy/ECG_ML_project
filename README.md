# aml_task2
General best practices/things to do: 
- connect your github account to PyCharm/your IDE (
  https://www.youtube.com/watch?v=KEfo6sHgVOc&ab_channel=ecodedev from 1:18 to about 4:28)
- the "master" branch is called main
- whenever you want to add/modify code:
    - update the main branch (on PyCharm: click on "main" in bottom right of your screen, click on main again and select update)
    - checkout from the main branch with branch name development_<your_name>
    - make your changes
    - commit your changes (in PyCharm: green arrow top right)
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

Try to comment your code and name things so that they're understandable, e.g. X_train_no_outliers vs X_WoO

If you feel like it, I recommend installing the Black tool for PyCharm: https://black.readthedocs.io/en/stable/integrations/editors.html
It cleans the code and makes it more legible