# Automatic Pavement Crack Detection for Japan, Czech Republic and India Datasets

Step 1: Download the complete repository 

Step 2: Install all the dependencies from requirements.txt
</br>
$ pip install -r requirements.txt

Step 3: Download 3 different models one each for every country [HERE](https://drive.google.com/drive/folders/1__WPNp2PFkZf0pbGwnCValb58Mc4DKYD)

Step 4: Create a new folder within the parent directory and place each of those models in it for testing.

# Testing

The train, test1 and test2 datasets can be downloaded from [HERE](https://github.com/sekilab/RoadDamageDetector).

Since, you already downloaded the models in Step 3. Proceed to Step 5-6 for Testing those models on test1 and test2 datasets.

# test1

Step 5: cd into Test1and2 folder and run the following python files separately by passing individual models (downloaded from Step 3) and by inserting the correct path of the models within these files:

trial.py: jpvl54.pt
</br>
trial_Czech.py:  jc66.pt
</br>
trial_india.py: last_india_new.pt

# test2

Step 6: Within the same folder Test1and2, you will need to pass different models into the following files to test on 'test2' dataset. Here are the following files and you will need to change the models as per the information detailed below:

trial2.py: jpvl54.pt
</br>
trial_Czech2.py:  jc66.pt
</br>
trial_india2.py: last_india_new.pt


