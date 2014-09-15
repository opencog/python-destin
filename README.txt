PythonDeSTIN is a repo for the development of Python DeSTIN (PyDeSTIN).
As a starting point: 
	-> here I have developed a Node and Learning Algorithm Classes
		- Just to get a sense og Learning Algorithm inside a node.
		  I put a LogisticRegression implemented in theano 
	-> LearningAlgorithm Class is placed as attribute of the Node class.
	-> LearningAlgorithm Object does the actual learning for the node.
The whole PyDeSTIN will have Four Classes: LearningAlgorithm, Node, Layer and Network
The Classes will be placed in a nested fashion as follows:
-> Network
     Layer
    	Node
	      LearningAlgorithm
Installation Instructions:
-> Clone the repo by running
    git clone https://github.com/tedyhabtegebrial/PythonDeSTIN.git
-> checkout to the UniformDeSTIN
    git checkout UniformDeSTIN
-> Downoad Cifar dataset for training and testing 
    Available at http://www.cs.toronto.edu/~kriz/cifar.html
    Download the python version
-> Edit loadData.py to modify the location of the Cifar directory/ where you placed the downloaded cifar dataset
-> run testDestin.py
_______________________________ Nessesary Libraries
See http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu for installing it on Ubuntu
You need to have python>2.73, Numpy and Scipy libraries installed
For the future versions theano will also be necessary so installing theano is optional at this time

Testing
Inorder to run the testWithSVM.py script and evaluate the classification accuracy on the cifar data set 
install the scikit-learn machine learning toolkit.
see installation instructions at http://scikit-learn.org/stable/install.html


Outlines for the Development of DeSTIN as a robust Spatio-Temporal Inference Engine
Taking into consideration points listed @: http://wiki.opencog.org/w/New_DeSTIN_Redesign_Proposal	
We will have explicit branches for A to D.

A) pure DeSTIN Framework: flexible enough to support different learning algorithms (Done)
		
B) Implemeting Online-NonNegative Sparse AutoEncoder in theano/or python (Done)
		
C) Implemeting Stable Incremental K Means Clustering in theano (In Progress)

D) a LeNet style CNN built using the general-purpose CNN layer ()
	(The theory may require revision)
	(How to make sense of the Complex and Simple cell like filters simulated in 	the CNN?)
	(Pooling is also an issue.....)

D) hybrid DeSTIN-CNN without feedback 


D) hybrid DeSTIN-CNN with feedback



Reading List For DeSTIN:

* http://web.eecs.utk.edu/~itamar/Papers/BICA2009.pdf
* http://www.ece.utk.edu/~itamar/Papers/BICA2011T.pdf
* http://web.eecs.utk.edu/~itamar/Papers/AI_MAG_2011.pdf
* http://research.microsoft.com/en-us/um/people/dongyu/nips2009/papers/Arel-DeSTIN_NIPS%20tpk2.pdf
* http://goertzel.org/Goertzel_AAAI11.pdf
* http://goertzel.org/DeSTIN_OpenCog_paper_v1.pdf
* http://goertzel.org/papers/DeSTIN_OpenCog_paper_v2.pdf
* http://www.springerlink.com/content/264p486742666751/fulltext.pdf
* http://goertzel.org/VisualAttention_AGI_11.pdf
* http://goertzel.org/Uniform_DeSTIN_paper.pdf
* http://goertzel.org/papers/Uniform_DeSTIN_paper_v2.pdf
* http://goertzel.org/papers/CogPrime_Overview_Paper.pdf

* Visit Dr. Itamar Arel's Machine Intelligence Lab at University of Tennessee at Knoxville. http://mil.engr.utk.edu/
