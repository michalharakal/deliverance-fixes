export MAVEN_OPTS="--add-opens java.base/java.nio=ALL-UNNAMED --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED    -Djava.library.path=/home/edward/deliverence/native/target/native-lib-only"
mvn io.teknek.deliverance:vibrant-maven-plugin:0.0.2-SNAPSHOT:generate

