# WARNING
### API CODE IS NOT UP TO DATE


# Cryptopunks Analysis
### Goal
Create 92 different CNN to identify each of the 92 possible traits of a crypto punk.

### Data.Obtaining.Traits
Data was gathered using the OpenSea API. Threading was used to speed up the process.

### Data.Obtaining.Images
The image of each punk is obtained by algorithmically cropping out each character from the original punk.png image which contains all of them. This image is recovered from the github page of the official cryptopunks project.

### Data.Cleaning
The gathered data was cleaned in a way to have binary values for each of the possible traits.

### Data.Building
The cleaned data is separated using an 80/20 split for Testing/Training. A training dataset is created for each trait. Each training dataset is comped of 16'000 observations. This is achieved by adequately oversampling the trait and no-trait punks. This allows to compensate for rare traits and create a balanced training dataset. All images are converted first converted to grayscale and then to an array representing each pixel's grayscale value.

### Network.Structure
The networks are all convolutional. Each has:<p>
<ol>
<li> Convolutional layer 1 to 32. Kernel 2x2. </li>
<li> Convolutional layer 32 to 64. Kernel 2x2. </li>
<li> Convolutional layer 64 to 128. Kernel 2x2. </li>
<li> Linear layer 128 to 512.</li>
<li> Linear layer 512 to 2.</li>
</ol>
The final result is then converted to a probability score by using the softmax algorithm.

### Network.Training
The networks are trained over 12 EPOCHS, with batches of 100. The Optimiser used is AdaGrad, its parameters are:
<ol>
<li> Learning Rate: 0.01 </li>
<li> Learning Rate Decay: 0.01 </li>
<li> Weight Decay: 0 </li>
<li> Initial Accumulator Value. </li>
<li> EPS: 1e-10.</li>
</ol>
This Optimiser worked best. The decaying learning Rate is used to minimise fluctuation over the minima found.

### Network.Training.Testing
Throughout training, we test at each step In & Out Accuracy & Loss. 10 observations are randomly sampled In & Out of the training of the training dataset and then tested.

### Performance.Metrics
The metric used to asses performance are:
<ol>
<li>Accuracy</li>
<li>Precision</li>
<li>Recall</li>
<li>F1</li>
</ol>

### Performance.Results
The results are overall good. Some traits remain difficult to asses, especially given their nature depending on color.