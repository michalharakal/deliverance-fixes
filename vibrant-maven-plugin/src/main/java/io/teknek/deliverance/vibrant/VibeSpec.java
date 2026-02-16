package io.teknek.deliverance.vibrant;

import org.apache.maven.plugins.annotations.Parameter;

import java.util.List;

public class VibeSpec {
    @Parameter(name = "id")
    String id;

    @Parameter(name = "userMessages")
    List<String> userMessages;

    public VibeSpec(){

    }

    public String getId() {
        return id;
    }
    public void setId(){

    }
    public void setId(String id) {
        this.id = id;
    }

    public List<String> getUserMessages() {
        return userMessages;
    }

    public void setUserMessages(List<String> userMessages) {
        this.userMessages = userMessages;
    }

    @Override
    public String toString() {
        return "VibeSpec{" +
                "id='" + id + '\'' +
                ", userMessages=" + userMessages +
                '}';
    }
}
