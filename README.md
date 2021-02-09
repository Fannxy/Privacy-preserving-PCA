# Privacy-preserving-PCA
Here we offer code for the privacy-preserving PCA algorithm.

## File Structure
As our work is based on the Privpy Framework, the cipher-text part should be run in the MPC platform. For reproducability, we offer the plain-text version in Plain-text for demonstration.

### Plain-text
**Care that you have to firstly donload the dataset to the Data folder for applications**

*	``plain-preprocessing.py`` shows the required algorithm for plain-text preprocessing
*	``jacobi.py and qr.py`` shows the plain-text version of our cipher-text algorithms, the result's correctness is within $1e-5$.
*	``application_iot.py and application_mooc.py`` shows the demo for data integration.
*  ``accuracy_test.py`` shows the correctness verification for our implementation with Scikit-learn.


### Cipher-text
The cipher-text algorithm required a Privpy (4,2)-secret sharing deployment, if you need please contact us~
