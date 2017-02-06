mvn clean package -P tank
#bash $SPARK_HOME/bin/spark-submit --class org.soic.eaclsh.KNNTester --master local[1] --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:ConcGCThread=1 -XX:+PrintGCDetails -XX:+PrintGCTimeStamps" --driver-memory 5g ./target/eaclsh-0.1-jar-with-dependencies.jar ./
bash $SPARK_HOME/bin/spark-submit --class org.soic.eaclsh.KNNTester --master local[2] ./target/eaclsh-0.1-jar-with-dependencies.jar ./
