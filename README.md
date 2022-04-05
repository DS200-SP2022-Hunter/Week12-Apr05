# Lab Assignment 11, Due on [Canvas](https://psu.instructure.com/courses/2174978/assignments/13906025), Apr. 13 at 11:59pm
## Implement a Naive Bayes Classifier

The main objective of today's lab is to use Bayes Theorem and some simplifying assumptions to create a classfier, known as a naive Bayes classifier, for predicting the category of wines based on several quantitative measurements.  


**Objective**:  Use the `wines` dataset from [Section 17.4](https://inferentialthinking.com/chapters/17/4/Implementing_the_Classifier.html) to train a naive Bayes classifier to predict the type of wine from quantitative measurements.  We'll make two simplifying assumptions:  conditional independence of features and the quantitative variables are all normally distributed.  

The basic idea of a naive Bayes classifier is described in [this document](https://github.com/DS200-SP2022-Hunter/Week12-Apr05/blob/main/NaiveBayes.pdf).

**Your assignment** is as follows:

1. Load the Jupyter notebook for [Section 17.4](https://inferentialthinking.com/chapters/17/4/Implementing_the_Classifier.html) of the textbook from GitHub as you've done in the past. You might want to change the `path_data` object so that it points to the URL where the dataset can be found: `http://personal.psu.edu/drh20/200DS/assets/data/` Also, for some of the steps below you may find it helpful to update your colab's version of the datascience library using this code before importing that library:
```
!pip install --upgrade datascience
```

2. As in Section 17.4.3, read the dataset from the `wines.csv` file as a `Table` object and give it the name `wines`.  Do NOT convert this dataset to a new one with only two classes of wine.  We will keep all three classes for this assignment.

3. As you will see from the naive Bayes classifier document(), you'll need the means and standard deviations of the quantitative variables for each wine class.  You can get them using the `group` method, which allows for an optional function to be used on the values in each group:
```
wineMeans = wine.group("Class", np.mean)
wineSDs = wine.group("Class", np.std)
```

12.  Finally, make sure that your Jupyter notebook only includes code and text that is relevant to this assignment.  For instance, if you have been completing this assignment by editing the original code from Section 13.2, make sure to delete the material that isn't relevant before turning in your work.

When you've completed this, you should select "Print" from the File menu, then save to pdf using this option.  The pdf file that you create in this way is the file that you should upload to Canvas for grading.  If you have trouble with this step, try selecting the "A3" paper size from the advanced options and making sure that your colab is zoomed out all the way (using ctrl-minus or command-minus).  As an alternative, you can create the pdf within your google drive space and then download it from there.  Here's a [Jupyter noteboook](https://github.com/DS200-SP2022-Hunter/Week11-Mar29/blob/main/convert_pdf.ipynb) shared by Xinyu Dou that creates the pdf within the google drive space (you may need to modify it depending on your directory names and the name of your lab file).

