# eac-lsh
scp -i ~/Desktop/vjw.pem target/eaclsh-0.1-jar-with-dependencies.jar hadoop@ec2-52-35-195-27.us-west-2.compute.amazonaws.com:~
hadoop fs -mkdir -p /home/hadoop/eaclsh/eac-lsh-master/dataset/activity/
unzip activityCleaned.data.zip
hadoop fs -put activityCleaned.data /home/hadoop/eaclsh/eac-lsh-master/dataset/activity/
spark-submit --class org.soic.eaclsh.KNNTester ./eaclsh-0.1-jar-with-dependencies.jar knnlsh
