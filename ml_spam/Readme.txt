 There are 2 codes provided in the folder.(I have provided all 3 classifications in both the programs, it was eaier that way!)
 code_for_input_file.py is the code written to work on the input.csv file.
python code_for_input_file.py should run the code and it takes the name of the csv file from the user.
The code is well documented but it is important to have the input.csv file contain more than 3972 urls. I have hard coded that.
The code_for_input_file.py prodices two png images for SVM and Logistic Regression respectively(2 features only).
The visualization for decision tree is saved as a pdf file.
The output for the classifiers for this code is stored in the files with the names having outut_input.csv appended to the model beung used.
There are certain things that can break the visualization part of the above code(dimensionality issues), change with caution!
The code_for_testing_file.py runs the classification on the testing_file.csv.
The prediction is stored in the 3  files as mentioned in the email sent by Sathya.
The program also calculates accuracy and prediction score but it depends on the label file you provide.
To run the code type the following in the command line: "python code_for_testing_file.csv"
The code requires 3 inputs from the user: name of the training file which has to be similar to the input.csv file provided, the name of the label file for checking accuracy and the name of teh testing file which would be testing.csv.
If you face nay issues with the label file or accuracy predictions just uncomment those lines in the code. I have mentioned which lines you would need to uncommnent in the code.
I have attached the images and the input.csv file on which I have worked here. You can use your own input.csv file but it would be nice if the number of rows is the same as the input file I have attached.
