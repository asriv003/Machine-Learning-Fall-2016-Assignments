This folder contains following.

- SVM.py : It was used by me to train and test the classifier using helding out test part. It also contains the code for learning curve, search grid and PCA reduction. I will suggest not using this one to train the data. This is only to show you how I train and tested the classifier for the project report.

- train.py : It should be used to train the data. by default I am asumming that the file for training data is 'handwriting.data' if you want to change the training data pass it as command argument while running the code.(for e.g : python train.py training.data). I have written down the code to retain the classifier using 'joblib' library. It will write down the classifier in the pickle format. This will take around 10 min (as per my system).

- test.py : It should be used to test the prediction of the classifier. But before running 'test.py' you must run 'train.py' so that classifier can be stored. Same as before currently I have assumed the default value fo file as 'handwriting.data' to change the filename pass it as command line argument.(for e.g : python test.py testing.data). After passing the file name data will be loaded and after applying PCA, Values will be predicted. After the prediction the values will be saved in a file. It takes about 6-8 minutes(as per my system).

- check_accuracy.py: It can be used to check the accuracy of the prediction made by the classifier. I am writing my prediction data to 'predicted_values.txt'. So to check the accuracy you must pass the file name where true Y values are stored. I am by default using 'handwriting.data' as correct true data file. I am also assuming that the first row will the true Y value and file will not have headers. This will just report the accuracy of the prediction by comparing with the true values. 

Both training.data, testing.data and trueY.data should be in the same folder as the code. I have not tried giving path of the file as command line argument so i am not sure that will work or not.

If you face any issue please mail me : asriv003@ucr.edu

Thank you!! :)