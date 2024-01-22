# HypnoTech - Sleep Stage Machine Learning Model

This machine learning model based on the random forest algorithm was developed at Rice University's 6th Datathon to to unlock critical insights into sleep patterns while alleviating the need for manual sleep stage classifications by physicians. By extracting influential features from six channels, HypnoTech can predict a patient’s current sleep stage.  

:trophy: Neurotech Track - 2nd

##How we built it

###Data Wrangling
We began by transforming the given 3D array into a 2D pandas dataframe, for easier data manipulation and wrangling. Next, we cleaned the data set of epochs that were not scored. Then, we visualized the distribution of the remaining sleep stages in a histogram. We noticed that there was a significantly higher number of waking sleep stages leading to an unbalanced dataset. To prevent the negative effects of training a machine learning model on an unbalanced dataset, we used undersampling to balance the dataset. We chose the ‘auto’ sampling strategy to avoid adding excessive duplicate epochs and to also avoid removing important characteristics.  