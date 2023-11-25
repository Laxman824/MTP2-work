# MTP2-work

The importance of accurate detection and extraction of data from tables in documents for making information accessible to all individuals, including those with visual impairments.
The thesis notes that while significant progress has been made in detecting and extracting data from bordered tables, borderless tables present unique challenges due to their lack of clear cell boundaries. 
Existing approaches such as the CascadeTabNet algorithm have shown low accuracy and slow performance in borderless table detection.

The thesis then provides an overview of previous work on table recognition and content extraction, including the limitations of existing approaches. The proposed solutions are then introduced, including the use of Layout parser and OpenCV to enhance the borderless table detection and content extraction, as well as the connecting of MTL-TabNet to replace the ineffective CascadeTabNet algorithm. 
The thesis also discusses the results achieved through these solutions and provides a conclusion.

The work related to machine learning in this project involves the use of a support vector machine (SVM) classifier to classify the cells in the table as either header cells or data cells. The classifier was trained on a dataset of labeled cells, with features such as cell size, position, and color used as input. Additionally, MTL-TabNet, a deep learning model, was used for image-based table recognition and information extraction from the image. These machine learning techniques were crucial in developing an effective solution for borderless table recognition and accessibility.

Overall, this thesis provides valuable insights into the use of computer vision techniques and machine learning algorithms for borderless table recognition and accessibility. The methodology involved several stages, including image preprocessing, table region detection, and cell classification. The use of machine learning algorithms such as SVM and MTL-TabNet
<img  alt="GIF" src="https://github.com/Laxman824/Laxman824/blob/main/Gifs/workflow.png?raw=true"  />
