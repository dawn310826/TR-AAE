# TR-AAE
We propose a way of deep collaborative filtering with Adversarial Autoencoders(AAES) for tag recommendation with a multinomial likelihood function. Our probability model is able to exceed the limited modeling capabilities of linear models and allows us to explore the complex co-occurrence relationships between tags and items on large-scale tag recommendation datasets. In addition, label smoothing is introduced to alleviate overfitting, and adjusted to tag recommendation scenario by modify the construction of the true probability.



**data**: citeulike-a and citeulike-t were used in the paper 'Collaborative Topic Regression with Social Regularization' [Wang, Chen and Li]. It was collected from CiteULike and Google Scholar. CiteULike allows users to create their own collections of articles. There are abstracts, titles, and tags for each article. Other information like authors, groups, posting time, and keywords is not used in this paper 'Collaborative Topic Regression with Social Regularization' [Wang, Chen and Li]. The details can be found at http://www.citeulike.ort/faq/data.adp. 



**load_data.py:** file used to create training, validation and testing data files. 

**AAE.py:** file to create our model TR-AAE

**train_AAE.py:** file to train our model TR-AAE

**metrics.py:** file to evaluate the performance of our model

