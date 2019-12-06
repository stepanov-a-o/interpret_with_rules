# Set-up Instructions 

The text below provides detailed set-up instructions for dependencies installation specifically for Mac OS (Catalina version 10.15.1), and it is also a good reference point for users of other operating systems.

For novice users, I recommend using Anaconda-Navigator (https://www.anaconda.com/), since it provides the most convenient library and environment management system. I also suggest using an older version of Anaconda (I run 1.8.7), since newer versions (e.g.  1.9.7) do not function with some of the required libraries (at least at the time when I am writing this). To download the previous version of Anaconda, please refer to https://repo.continuum.io/archive/. 

### To use the code, you will need to create an environment satisfying the following requirements:

1. Python 3.6. I use version 3.6.8 from Anaconda distribution.

2. Numpy, scipy and scikit-learn == 0.19.1. Those libraries are pre-installed in Anaconda. 

3. Xcode Command Line Tools. Run 'xcode-select --install' in the terminal to install Command Line Tools. If you have questions, this forum discussion can help https://stackoverflow.com/questions/52522565/git-is-not-working-after-macos-update-xcrun-error-invalid-active-developer-pa.

4. Pytorch library (torch == 0.4.0). You can use "conda install pytorch=0.4.0 -c pytorch" command in the terminal. 

5. Javabridge library. Note that you need to install Xcode Command Line Tools first and then install javabridge. Use 'pip install javabridge' to install the library. I had some issues with that library. The version with following specifications works for me (see below).

    - java version "11.0.4" 2019-07-16 LTS
    - Java(TM) SE Runtime Environment 18.9 (build 11.0.4+10-LTS)
    - Java HotSpot(TM) 64-Bit Server VM 18.9 (build 11.0.4+10-LTS, mixed mode)
    
    To get specifications, run 'java -version' in the terminal. For all questions and troubleshooting please refer to                   https://github.com/LeeKamentsky/python-javabridge. This page can also be useful to set JAVA_HOME on MAC                      http://www.sajeconsultants.com/how-to-set-java_home-on-mac-os-x/. 

6. Python weka wrapper3 library. You can use 'python-weka-wrapper3' command in the terminal. For WEKA troubleshooting, please refer to https://fracpete.github.io/python-weka-wrapper/troubleshooting.html.

7. Fuzzy Unordered Rule Induction (FURIA) algorithm. FURIA is not part of the python-weka-wrapper standard distribution. Download FURIA from http://weka.sourceforge.net/packageMetaData/fuzzyUnorderedRuleInduction/index.html. Note that when you download and unzip the folder, you will need to find 'fuzzyUnorderedRuleInduction.jar' file and copy it to 'lib' folder within the path for weka. In my case, the path to the folder is /Applications/anaconda2/envs/py36-dev/lib/python3.6/site-packages/weka. 
