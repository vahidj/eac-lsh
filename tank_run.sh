mvn clean package -P tank
bash $SPARK_HOME/bin/spark-submit --class org.soic.eac.KNNTester --master local[2] ./target/eac-0.0.1-SNAPSHOT.jar ./
