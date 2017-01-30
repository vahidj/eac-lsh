mvn clean package -P home
bash ~/Work/spark/spark-1.6.0-bin-hadoop2.6/bin/spark-submit --class org.soic.eaclsh.KNNTester --master local[2] ./target/eaclsh-0.1-jar-with-dependencies.jar ./
