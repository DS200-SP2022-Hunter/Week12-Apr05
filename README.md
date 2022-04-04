# Lab Assignment 11, Due on [Canvas](https://psu.instructure.com/courses/2174978/assignments/13906025), Apr. 13 at 11:59pm
## Implement a Naive Bayes Classifier

The main objective of today's lab is to use Bayes Theorem and some simplifying assumptions to create a classfier, known as a naive Bayes classifier, for predicting the category of wines based on several quantitative measurements.  


**Objective**:  Use the `wines` dataset from [Section 17.4](https://inferentialthinking.com/chapters/17/4/Implementing_the_Classifier.html) to train a naive Bayes classifier to predict ...  We'll make two simplifying assumptions:  independence of features and the quantitative variables are all normally distributed.

**Your assignment** is as follows:

1. Load the Jupyter notebook for [Section 17.1](https://inferentialthinking.com/chapters/17/1/Nearest_Neighbors.html) of the textbook from GitHub as you've done in the past. You might want to change the `path_data` object so that it points to the URL where the dataset can be found:  `http://personal.psu.edu/drh20/200DS/assets/data/` Also, for some of the steps below you may find it helpful to update your colab's version of the datascience library using this code before importing that library:
```
!pip install --upgrade datascience
```

2. As in the first block of code in Section 17.1.1, read the dataset from the `ckd.csv` file as a `Table` object and give it the name `ckd`, for chronic kidney disease . 

3. We only need a 3-column subset of this dataset, so (again, as in Section 17.1) rename the `ckd` object as follows:
```
ckd = ckd.select('Class', 'Hemoglobin', 'White Blood Cell Count')
```
4. Produce a scatterplot with hemoglobin on the horizontal axis and white blood cell count on the vertical axis, with the points labeled by color according to their values of `Class` (1 for CKD, 0 for no CKD). Use the last block of code in Section 17.1.1 as a model.  

5. In your scatterplot, you should see the cases without CKD in gold in the lower right corner and the cases with CKD in dark blue mostly on the left.  Suppose you wanted to draw a stright line connecting the point (x1, 5000) to the point (x2, 25000) that mostly separates the two clusters of points.  What values of x1 and x2 would you use?  Using your values, add this line to your scatterplot using the following code just after the `scatter` method:
```
plots.plot([x1, x2], [5000,25000]);  # Be sure to substitute actual values for x1 and x2 here
```

6. Study the function called `lw_mse` in Section 15.3.3.1.  Notice that the final value returned equals the mean of the values `(y - fitted) ** 2`.  While this is an appropriate for linear regression, we will replace it with a specially tailored logistic regression function when `y` can only take the values 0 and 1, namely, `np.log(1 + np.exp(fitted)) - y * fitted`.  Here is a function that can therefore replace `lw_mse`:
```
def ckd_logistic(a, b, c):
  Hem = ckd.column('Hemoglobin')
  WBC = ckd.column('White Blood Cell Count')
  y = ckd.column('Class')
  linear = a*Hem + b*WBC + c
  return np.mean(np.log(1+np.exp(linear)) - y*linear)
```
Notice that we have used `linear` in place of `fitted` above.  This is a semantic choice; in linear regression, the linear function is the same as the fitted value, but in logistic regression the linear function is NOT the fitted value so the use of `fitted` might be confusing.

7. Use the `minimize` function with the `ckd_logistic` to find the values of `a`, `b`, and `c` that minimize the value of `ckd_logistic`.  Report these three values in your output.

8. The logistic function maps the value of `linear` to the probability given by `np.exp(linear) / (1 + np.exp(linear))`.  This means that when `linear` is zero, the probability equals 1/2.  Similarly, the probably is greater than 1/2 or less than 1/2 when `linear` is greater than 0 or less than 0, respectively.  

9. According to Step 8, we can find all the points `(Hem, WBC)` where our logistic regression predicts a probability of 1/2 for having CKD:  They are exactly those points for which `linear=0` using the values from Step 7.  These points fall on a straight line in the scatterplot.  Going back to Step 5, solve for the values of x1 and x2 such that (x1, 5000) and (x2, 25000) are on this line, then recreate the scatterplot with this line.  

10. The line you created in Step 9 gives us what is called a "hard classifier":  All points on one side are classified as CKD, and all points on the other are classified as non-CKD.  However, logistic regression actually provides more than a hard classifier; it gives each `(Hem, WBC)` point a probability of CKD according to the formula in Step 8.  As the final step in your lab this week, calculate the probability associated with the point `(Hem=12, WBC=7500)`.
 
11. If you're curious where the expression `np.log(1 + np.exp(fitted)) - y * fitted` in Step 6 comes from, check out this [short writeup](https://github.com/DS200-SP2022-Hunter/Week11-Mar29/blob/main/LogisticRegression.pdf).
  
12.  Finally, make sure that your Jupyter notebook only includes code and text that is relevant to this assignment.  For instance, if you have been completing this assignment by editing the original code from Section 13.2, make sure to delete the material that isn't relevant before turning in your work.

When you've completed this, you should select "Print" from the File menu, then save to pdf using this option.  The pdf file that you create in this way is the file that you should upload to Canvas for grading.  We have found that if you can select the "A3" paper size from the advanced options, this seems to solve the problems that are sometimes encountered in this step.


