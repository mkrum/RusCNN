This was a project I did for my data science class over the summer of 2016. This is a convolutional neural network that attempts to distinguish between three different artistic styles: realism, surrealism, and abstract. Here are the results I achieved:

![alt tag](https://raw.githubusercontent.com/mkrum/RusCNN/images/models.png)
![alt tag](https://raw.githubusercontent.com/mkrum/RusCNN/images/compare.png)

I used two models: a classical LeNet Model and a model that used inception layers. To run these, use:

  ./tfcnn.py S R A results.txt  (LeNet version)
  or
  ./incepdim.py S R A results.txt (Inception Layer version)

where S, R, and A are 0 or 1 denoting whether or not you wish to include (S)urrealism, (R)ealism, or (A)bstract works. The results.txt file is just where it will store the results. 

I would not recommend running this on a CPU as it will take an inordinate to complete.
