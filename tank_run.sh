mvn clean package -P tank
bash $SPARK_HOME/bin/spark-submit --class org.soic.eaclsh.KNNTester --master local[2] ./target/eaclsh-0.1-jar-with-dependencies.jar ./
